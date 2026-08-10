"""
Per-case scoring against ground truth.

Paper reference: this is the concrete implementation of what Section 4.6
("Metrics") in the requested reproducible-methodology outline must define
precisely for each module — exact field-level correctness criteria rather
than an unstated "accuracy" number.
"""

from __future__ import annotations

NUMERIC_TOLERANCE = 0.05  # relative tolerance for numeric fields


def _numeric_ok(pred, true, tol=NUMERIC_TOLERANCE) -> bool:
    if true == 0:
        return abs(pred - true) < 1e-9
    return abs(pred - true) / abs(true) <= tol


def score_case(module: str, output: dict | None, ground_truth: dict) -> float:
    """
    Returns a score in [0, 1]: the fraction of ground-truth fields the
    model output matched, under the module-specific field comparators
    below. Returns 0.0 if output is None (e.g. BLOCKED/ESCALATE cases).
    """
    if output is None:
        return 0.0

    if module == "M1_SENTIMENT":
        checks = [
            _numeric_ok(output.get("signal", 0.0), ground_truth["signal"], tol=0.5),
            output.get("direction") == ground_truth["direction"],
            output.get("alert_fired") == ground_truth["alert_fired"],
        ]
    elif module == "M2_EARNINGS":
        checks = [
            _numeric_ok(output.get("revenue_gaap", 0.0), ground_truth["revenue_gaap"]),
            _numeric_ok(output.get("eps_non_gaap", 0.0), ground_truth["eps_non_gaap"]),
            _numeric_ok(output.get("guidance_low", 0.0), ground_truth["guidance_low"]),
            _numeric_ok(output.get("guidance_high", 0.0), ground_truth["guidance_high"]),
            output.get("full_document_processed") == ground_truth["full_document_processed"],
        ]
    elif module == "M3_RISK":
        checks = [
            _numeric_ok(output.get("var99", 0.0), ground_truth["var99"], tol=0.15),
            _numeric_ok(output.get("cvar99", 0.0), ground_truth["cvar99"], tol=0.15),
            bool(output.get("ips_violations")) == bool(ground_truth["ips_violations"]),
            output.get("advisor_review_required") == ground_truth["advisor_review_required"],
        ]
    elif module == "M4_COMPLIANCE":
        pred_flag = output.get("attorney_review_flag")
        checks = [
            pred_flag == ground_truth["attorney_review_flag"],
            len(output.get("findings", [])) > 0 if ground_truth["findings"] else len(output.get("findings", [])) == 0,
        ]
    else:
        raise ValueError(f"Unknown module '{module}'")

    return sum(bool(c) for c in checks) / len(checks)
