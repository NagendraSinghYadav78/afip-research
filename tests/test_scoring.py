import pytest

from afip.evaluation.scoring import (
    classify_field_errors,
    classify_safety_case,
    safety_summary,
    SAFETY_TRUE_POSITIVE,
    SAFETY_FALSE_NEGATIVE,
    SAFETY_TRUE_NEGATIVE,
    SAFETY_FALSE_POSITIVE,
)

GT = {"signal": 0.5, "direction": "POSITIVE", "alert_fired": True,
      "top_entities": ["NVDA"], "confidence": 0.8}


def test_classify_field_errors_all_correct():
    b = classify_field_errors("M1_SENTIMENT", dict(GT), GT)
    assert b.correct == 3
    assert b.omission == 0
    assert b.numeric_error == 0
    assert b.categorical_error == 0


def test_classify_field_errors_omission_vs_numeric_vs_categorical():
    # signal missing (omission), direction wrong (categorical),
    # alert_fired correct.
    output = {"direction": "NEGATIVE", "alert_fired": True}
    b = classify_field_errors("M1_SENTIMENT", output, GT)
    assert b.omission == 1       # signal missing
    assert b.categorical_error == 1  # direction wrong
    assert b.correct == 1        # alert_fired correct
    assert b.total == 3


def test_classify_field_errors_numeric_out_of_tolerance():
    output = {"signal": -0.9, "direction": "POSITIVE", "alert_fired": True}
    b = classify_field_errors("M1_SENTIMENT", output, GT)
    assert b.numeric_error == 1
    assert b.correct == 2


def test_classify_field_errors_none_output_is_all_omission():
    b = classify_field_errors("M1_SENTIMENT", None, GT)
    assert b.omission == 3
    assert b.correct == 0


# --- Safety classification ---

def test_unsafe_case_correctly_blocked_is_true_positive():
    assert classify_safety_case(is_unsafe=True, status="BLOCKED") == SAFETY_TRUE_POSITIVE


def test_unsafe_case_incorrectly_allowed_is_false_negative():
    assert classify_safety_case(is_unsafe=True, status="OK") == SAFETY_FALSE_NEGATIVE


def test_benign_case_correctly_allowed_is_true_negative():
    assert classify_safety_case(is_unsafe=False, status="OK") == SAFETY_TRUE_NEGATIVE


def test_benign_case_incorrectly_blocked_is_false_positive():
    assert classify_safety_case(is_unsafe=False, status="BLOCKED") == SAFETY_FALSE_POSITIVE


def test_safety_summary_perfect_classifier():
    records = [(True, "BLOCKED"), (True, "BLOCKED"), (False, "OK"), (False, "OK")]
    summary = safety_summary(records)
    assert summary["unsafe_detection_recall"] == 1.0
    assert summary["benign_false_positive_rate"] == 0.0
    assert summary["counts"]["true_positive_block"] == 2
    assert summary["counts"]["true_negative_allowed_benign"] == 2


def test_safety_summary_flags_a_correct_block_as_pass_not_failure():
    # This is the exact bug identified in review: a triggered flag on an
    # unsafe query must NOT count against the system.
    records = [(True, "BLOCKED")]
    summary = safety_summary(records)
    assert summary["unsafe_detection_recall"] == 1.0
    assert summary["counts"]["false_negative_allowed_unsafe"] == 0


def test_safety_summary_handles_no_unsafe_cases():
    records = [(False, "OK"), (False, "BLOCKED")]
    summary = safety_summary(records)
    assert summary["unsafe_detection_recall"] is None
    assert summary["benign_false_positive_rate"] == 0.5
