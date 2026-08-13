from pathlib import Path
import tempfile

from afip.evaluation.harness import load_cases, run_evaluation, write_results
from afip.evaluation.scoring import score_case
from afip.clients.mock_client import MockClient

REPO_ROOT = Path(__file__).parent.parent


def test_load_sample_cases():
    cases = load_cases(REPO_ROOT / "data" / "sample_cases.jsonl")
    assert len(cases) == 11  # 9 benign + 2 unsafe demo cases
    modules = {c["module"] for c in cases}
    assert modules == {"M1_SENTIMENT", "M2_EARNINGS", "M3_RISK", "M4_COMPLIANCE"}
    unsafe = [c for c in cases if c.get("is_unsafe")]
    assert len(unsafe) == 2


def test_score_case_perfect_match_is_one():
    gt = {"signal": 0.5, "direction": "POSITIVE", "alert_fired": True,
          "top_entities": ["NVDA"], "confidence": 0.8}
    assert score_case("M1_SENTIMENT", dict(gt), gt) == 1.0


def test_score_case_none_output_is_zero():
    gt = {"signal": 0.5, "direction": "POSITIVE", "alert_fired": True,
          "top_entities": ["NVDA"], "confidence": 0.8}
    assert score_case("M1_SENTIMENT", None, gt) == 0.0


def test_run_evaluation_end_to_end_zero_error_backbone_beats_high_error():
    cases = load_cases(REPO_ROOT / "data" / "sample_cases.jsonl")
    backbones = {
        "perfect": MockClient("perfect", seed=1, error_rate=0.0, schema_violation_rate=0.0),
        "noisy": MockClient("noisy", seed=2, error_rate=0.8, schema_violation_rate=0.0),
    }
    results = run_evaluation(
        cases=cases, backbones=backbones, focal_backbone="perfect",
        audit_log_path=None, n_resamples=500, seed=3,
    )
    assert results["per_backbone"]["perfect"]["mean_score"] == 1.0
    assert results["per_backbone"]["perfect"]["mean_score"] > results["per_backbone"]["noisy"]["mean_score"]
    row = results["significance"][0]
    assert row["backbone"] == "noisy"
    assert row["outcome"] == "win"  # perfect backbone should win significantly
    assert row["significant"] is True


def test_write_results_creates_expected_files():
    cases = load_cases(REPO_ROOT / "data" / "sample_cases.jsonl")
    backbones = {
        "a": MockClient("a", seed=1, error_rate=0.1),
        "b": MockClient("b", seed=2, error_rate=0.1),
    }
    results = run_evaluation(cases=cases, backbones=backbones, focal_backbone="a",
                              audit_log_path=None, n_resamples=200, seed=1)
    with tempfile.TemporaryDirectory() as tmp:
        write_results(results, out_dir=tmp)
        assert (Path(tmp) / "results.csv").exists()
        assert (Path(tmp) / "summary.json").exists()


def test_run_evaluation_single_backbone_pilot_does_not_crash():
    """
    Regression test: a single-backbone pilot run (the real Groq/Llama use
    case) must not raise, and must clearly flag itself as a pilot with no
    significance testing performed.
    """
    cases = load_cases(REPO_ROOT / "data" / "sample_cases.jsonl")
    backbones = {"solo": MockClient("solo", seed=1, error_rate=0.1)}
    results = run_evaluation(cases=cases, backbones=backbones, focal_backbone="solo",
                              audit_log_path=None, n_resamples=200, seed=1)
    assert results["single_backbone_pilot"] is True
    assert results["significance"] == []
    assert results["per_backbone"]["solo"]["n"] == len(cases)


def test_schema_fidelity_and_latency_exclude_safety_blocked_cases():
    """
    Regression test for the metric-definition bug flagged in external review:
    a correctly-blocked unsafe case must not be counted as a schema failure,
    and its 0.0s latency must not be averaged into inference latency.
    """
    cases = load_cases(REPO_ROOT / "data" / "sample_cases.jsonl")
    unsafe_cases = [c for c in cases if c.get("is_unsafe")]
    assert len(unsafe_cases) == 2  # sanity check on the fixture itself

    backbones = {"perfect": MockClient("perfect", seed=1, error_rate=0.0, schema_violation_rate=0.0)}
    results = run_evaluation(cases=cases, backbones=backbones, focal_backbone="perfect",
                              audit_log_path=None, n_resamples=200, seed=1)

    stats = results["per_backbone"]["perfect"]
    n_total = len(cases)
    n_blocked = len(unsafe_cases)  # MockClient never sees these; Step 1 blocks first

    assert stats["n"] == n_total
    assert stats["n_reached_inference"] == n_total - n_blocked
    # A perfectly-behaving backbone should show full schema fidelity on the
    # cases that actually reached inference, not be dragged down by blocks.
    assert stats["schema_fidelity"] == 1.0
    assert stats["safety_block_rate"] == round(n_blocked / n_total, 4)
    # Latency stats must be computed only over inference calls (non-zero).
    assert stats["mean_latency_seconds"] > 0
    assert stats["median_latency_seconds"] > 0
    assert stats["p90_latency_seconds"] >= stats["median_latency_seconds"]
