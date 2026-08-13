"""
Evaluation harness.

Paper reference: Section 6 ("We executed the entire n=1020 AFIP test suite
with each of the five frontier LLMs as the backbone in place of Claude by
keeping AFIP wrapper, MODULE_SCHEMA definitions, FINANCIAL_CONSTITUTION_PROMPT
and all evaluation harnesses identical") and Table 6A / Table 7.

This is that harness, runnable end-to-end. It:
  1. loads a labelled case set (data/sample_cases.jsonl by default — an
     11-case ORIGINAL DEMO subset, not the paper's planned n~1020 dataset),
  2. runs afip.algorithms.master_orchestration.run for every (case, backbone)
     pair, keeping the pipeline identical across backbones,
  3. scores each output against ground truth (afip.evaluation.scoring),
  4. runs the paired bootstrap + Bonferroni statistics
     (afip.evaluation.stats) comparing a focal backbone against every
     other backbone,
  5. writes results/results.csv (per-case) and results/summary.json
     (per-backbone + significance table), in the same shape as the
     paper's Table 6A / Table 7 so the mapping from code output to
     manuscript table is direct.

IMPORTANT: with the default MockClient backbones this produces SYNTHETIC
data for pipeline validation only (see afip.clients.mock_client). Pass
real LLMClient instances (afip.clients.anthropic_client.AnthropicClient,
etc.) to reproduce the paper's actual claims.
"""

from __future__ import annotations
import csv
import json
from pathlib import Path

import numpy as np

from ..algorithms.master_orchestration import run as run_algorithm_1, AuditLog
from ..clients.base import LLMClient
from .scoring import score_case, safety_summary
from .stats import paired_bootstrap_test, bonferroni_correct, summarize_family


def load_cases(path: str | Path) -> list[dict]:
    cases = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                cases.append(json.loads(line))
    return cases


def run_evaluation(
    cases: list[dict],
    backbones: dict[str, LLMClient],
    focal_backbone: str,
    audit_log_path: str | Path | None = "results/audit_log.jsonl",
    n_resamples: int = 10_000,
    seed: int = 42,
    alpha: float = 0.05,
) -> dict:
    """
    Returns a results dict with:
      - "per_case": list of {case_id, module, backbone, status, schema_valid,
                              attempts, latency_seconds, score}
      - "per_backbone": {backbone: {mean_score, schema_fidelity, mean_latency, n}}
      - "significance": Bonferroni-corrected paired-bootstrap comparison of
        focal_backbone vs every other backbone, family size = n other backbones
      - "family_summary": win/loss/nonsignificant counts (paper-style prose)
    """
    if focal_backbone not in backbones:
        raise ValueError(f"focal_backbone '{focal_backbone}' not in backbones dict")

    audit_log = AuditLog(audit_log_path) if audit_log_path else None

    per_case_rows = []
    scores: dict[str, dict[str, float]] = {name: {} for name in backbones}
    safety_records: dict[str, list[tuple[bool, str]]] = {name: [] for name in backbones}

    for case in cases:
        is_unsafe = bool(case.get("is_unsafe", False))
        for backbone_name, client in backbones.items():
            result = run_algorithm_1(
                case_id=f"{case['case_id']}::{backbone_name}",
                query=case["query"],
                module=case["module"],
                client=client,
                document=case.get("document"),
                knowledge_base=case.get("knowledge_base"),
                ground_truth=case.get("ground_truth"),
                audit_log=audit_log,
            )
            score = score_case(case["module"], result.output, case["ground_truth"])
            scores[backbone_name][case["case_id"]] = score
            safety_records[backbone_name].append((is_unsafe, result.status))
            per_case_rows.append({
                "case_id": case["case_id"],
                "module": case["module"],
                "backbone": backbone_name,
                "is_unsafe": is_unsafe,
                "status": result.status,
                "schema_valid": result.schema_valid,
                "attempts": result.attempts,
                "latency_seconds": round(result.latency_seconds, 4),
                "score": round(score, 4),
            })

    # Per-backbone aggregate stats.
    #
    # IMPORTANT METRIC-DEFINITION NOTE (fixed after external review): a
    # safety-gate block is correct system behavior, not a schema failure or
    # a zero-latency inference call. Earlier versions of this function
    # averaged schema_valid and latency_seconds over ALL cases, including
    # BLOCKED ones — that penalizes correct safety behavior as if it were a
    # schema defect, and deflates the inference-latency estimate with
    # spurious 0.0s entries. Both are now computed only over cases that
    # actually reached inference (status != "BLOCKED"); safety-gate
    # behavior is reported separately via `safety_summary` /
    # `safety_block_rate`.
    per_backbone = {}
    for name in backbones:
        rows = [r for r in per_case_rows if r["backbone"] == name]
        benign_rows = [r for r in rows if not r["is_unsafe"]]
        inference_rows = [r for r in rows if r["status"] != "BLOCKED"]
        latencies = [r["latency_seconds"] for r in inference_rows]

        per_backbone[name] = {
            "n": len(rows),
            "n_reached_inference": len(inference_rows),
            "mean_score": round(float(np.mean([r["score"] for r in benign_rows])), 4) if benign_rows else None,
            "schema_fidelity": (
                round(float(np.mean([r["schema_valid"] for r in inference_rows])), 4)
                if inference_rows else None
            ),
            "safety_block_rate": round(1 - len(inference_rows) / len(rows), 4) if rows else None,
            "mean_latency_seconds": round(float(np.mean(latencies)), 4) if latencies else None,
            "median_latency_seconds": round(float(np.median(latencies)), 4) if latencies else None,
            "p90_latency_seconds": round(float(np.percentile(latencies, 90)), 4) if latencies else None,
            "safety": safety_summary(safety_records[name]),
        }

    # Paired bootstrap of focal_backbone vs every other backbone (Table 6A/7 style).
    # A pilot run with only one backbone (nothing to compare against) is a
    # legitimate use case (see the real Groq/Llama pilot in the paper's
    # Section 4.8/4.9) — skip significance testing rather than raising, and
    # say so explicitly in the result.
    other_backbones = [name for name in backbones if name != focal_backbone]

    if not other_backbones:
        significance = []
        family_summary = {"wins": 0, "losses": 0, "nonsignificant": 0}
        corrected_alpha = None
    else:
        case_ids = [c["case_id"] for c in cases]
        focal_scores = np.array([scores[focal_backbone][cid] for cid in case_ids])

        raw_p = {}
        diffs = {}
        for name in other_backbones:
            other_scores = np.array([scores[name][cid] for cid in case_ids])
            boot = paired_bootstrap_test(focal_scores, other_scores, n_resamples=n_resamples, seed=seed)
            raw_p[name] = boot.p_value
            diffs[name] = boot.observed_diff

        corrected_alpha, corrected = bonferroni_correct(raw_p, alpha=alpha)
        directions = {name: (1 if diffs[name] >= 0 else -1) for name in raw_p}
        family_summary_full = summarize_family(corrected, directions)
        family_summary = {k: v for k, v in family_summary_full.items() if k != "detail"}

        significance = [
            {
                "backbone": r.label,
                "observed_diff": round(diffs[r.label], 4),
                "p_raw": round(r.p_raw, 5),
                "bonferroni_alpha": round(r.p_threshold, 6),
                "significant": r.significant,
                "outcome": family_summary_full["detail"][r.label],
            }
            for r in corrected
        ]

    return {
        "per_case": per_case_rows,
        "per_backbone": per_backbone,
        "significance": significance,
        "family_summary": family_summary,
        "focal_backbone": focal_backbone,
        "n_resamples": n_resamples,
        "alpha": alpha,
        "audit_log_verified": audit_log.verify() if audit_log else None,
        "single_backbone_pilot": not other_backbones,
    }


def write_results(results: dict, out_dir: str | Path = "results") -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "results.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results["per_case"][0].keys()))
        writer.writeheader()
        writer.writerows(results["per_case"])

    summary = {k: v for k, v in results.items() if k != "per_case"}
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
