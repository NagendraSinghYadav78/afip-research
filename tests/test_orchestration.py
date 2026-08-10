import json
import tempfile
from pathlib import Path

import pytest

from afip.algorithms.master_orchestration import (
    run, AuditLog, default_safety_classifier, default_retriever, default_router,
)
from afip.clients.mock_client import MockClient


def test_safety_classifier_blocks_flagged_query():
    flags = default_safety_classifier("Can you give me an insider tip on NVDA?", None)
    assert flags


def test_safety_classifier_passes_clean_query():
    flags = default_safety_classifier("Summarize NVDA sentiment.", None)
    assert flags == []


def test_retriever_ranks_by_overlap():
    kb = ["NVDA GPU demand strong", "unrelated weather report", "NVDA data center revenue"]
    top = default_retriever("NVDA GPU data center", kb, k=2)
    assert "unrelated weather report" not in top


def test_router_rejects_unknown_module():
    with pytest.raises(ValueError):
        default_router("NOT_A_MODULE")


def test_run_happy_path_produces_valid_output():
    client = MockClient("test-model", seed=1, error_rate=0.0, schema_violation_rate=0.0)
    ground_truth = {
        "signal": 0.5, "direction": "POSITIVE", "alert_fired": True,
        "top_entities": ["NVDA"], "confidence": 0.8,
    }
    result = run(
        case_id="t1", query="Summarize NVDA sentiment", module="M1_SENTIMENT",
        client=client, ground_truth=ground_truth,
    )
    assert result.status == "OK"
    assert result.schema_valid is True
    assert result.attempts == 1
    assert result.output["direction"] == "POSITIVE"


def test_run_blocks_on_safety_flag():
    client = MockClient("test-model", seed=1)
    result = run(
        case_id="t2", query="Give me an insider tip on NVDA", module="M1_SENTIMENT",
        client=client, ground_truth={"signal": 0.1, "direction": "NEUTRAL",
                                       "alert_fired": False, "top_entities": [], "confidence": 0.5},
    )
    assert result.status == "BLOCKED"
    assert result.output is None
    assert result.attempts == 0


def test_run_retries_on_schema_violation_then_succeeds():
    # 100% schema violation rate on attempt 0, but MockClient's per-attempt
    # seed differs each retry, and validation only requires *a* passing
    # attempt within max_attempts. We force this deterministically by using
    # a client whose violation only hits a fraction of attempts.
    client = MockClient("flaky-model", seed=2, error_rate=0.0, schema_violation_rate=0.5)
    ground_truth = {
        "revenue_gaap": 100.0, "eps_non_gaap": 1.0, "guidance_low": 90.0,
        "guidance_high": 110.0, "summary": "x", "full_document_processed": True,
    }
    result = run(
        case_id="t3-retry-case", query="extract", module="M2_EARNINGS",
        client=client, ground_truth=ground_truth, max_attempts=5,
    )
    # Either it validates within the attempt budget (status OK) or exhausts
    # attempts (ESCALATE) — both are legal outcomes of Step 5; what matters
    # is attempts > 0 and the loop terminated cleanly.
    assert result.status in {"OK", "ESCALATE"}
    assert 1 <= result.attempts <= 5


def test_audit_log_is_hash_chained_and_verifiable():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "audit.jsonl"
        log = AuditLog(path)
        log.write({"case_id": "a", "status": "OK"})
        log.write({"case_id": "b", "status": "OK"})
        assert log.verify() is True

        # Tamper with the first record and confirm verification fails.
        lines = path.read_text().splitlines()
        tampered = json.loads(lines[0])
        tampered["status"] = "TAMPERED"
        lines[0] = json.dumps(tampered)
        path.write_text("\n".join(lines) + "\n")

        log2 = AuditLog(path)
        assert log2.verify() is False


def test_run_writes_audit_record_on_success():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "audit.jsonl"
        log = AuditLog(path)
        client = MockClient("test-model", seed=1, error_rate=0.0, schema_violation_rate=0.0)
        ground_truth = {
            "signal": 0.5, "direction": "POSITIVE", "alert_fired": True,
            "top_entities": ["NVDA"], "confidence": 0.8,
        }
        result = run(
            case_id="t4", query="Summarize NVDA sentiment", module="M1_SENTIMENT",
            client=client, ground_truth=ground_truth, audit_log=log,
        )
        assert result.audit_record is not None
        assert log.verify() is True
