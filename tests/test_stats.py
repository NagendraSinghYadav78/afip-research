import numpy as np
import pytest

from afip.evaluation.stats import paired_bootstrap_test, bonferroni_correct, summarize_family


def test_bootstrap_identical_distributions_not_significant():
    rng = np.random.default_rng(0)
    a = rng.normal(0.8, 0.1, size=200)
    b = a.copy()  # identical -> true diff is exactly zero
    result = paired_bootstrap_test(a, b, n_resamples=2000, seed=1)
    assert result.observed_diff == pytest.approx(0.0, abs=1e-9)
    assert result.p_value > 0.05


def test_bootstrap_clearly_different_distributions_significant():
    rng = np.random.default_rng(0)
    a = rng.normal(0.9, 0.05, size=200)
    b = rng.normal(0.5, 0.05, size=200)
    result = paired_bootstrap_test(a, b, n_resamples=2000, seed=1)
    assert result.observed_diff > 0.3
    assert result.p_value < 0.001


def test_bootstrap_requires_equal_length():
    with pytest.raises(ValueError):
        paired_bootstrap_test(np.array([1.0, 2.0]), np.array([1.0]))


def test_bootstrap_reproducible_with_same_seed():
    rng = np.random.default_rng(0)
    a = rng.normal(0.7, 0.1, size=50)
    b = rng.normal(0.6, 0.1, size=50)
    r1 = paired_bootstrap_test(a, b, n_resamples=1000, seed=7)
    r2 = paired_bootstrap_test(a, b, n_resamples=1000, seed=7)
    assert r1.p_value == r2.p_value
    assert r1.ci_low == r2.ci_low


def test_bonferroni_correction_scales_with_family_size():
    p_values = {"m1": 0.01, "m2": 0.02, "m3": 0.20}
    alpha, results = bonferroni_correct(p_values, alpha=0.05)
    assert alpha == pytest.approx(0.05 / 3)  # 0.016667
    sig = {r.label: r.significant for r in results}
    assert sig["m1"] is True   # 0.01  < 0.016667 -> significant
    assert sig["m2"] is False  # 0.02  > 0.016667 -> not significant
    assert sig["m3"] is False  # 0.20  > 0.016667 -> not significant

    # A p-value clearly below the corrected threshold should be significant.
    p_values2 = {"m1": 0.001, "m2": 0.02, "m3": 0.20}
    _, results2 = bonferroni_correct(p_values2, alpha=0.05)
    assert {r.label: r.significant for r in results2}["m1"] is True


def test_bonferroni_rejects_empty_family():
    with pytest.raises(ValueError):
        bonferroni_correct({}, alpha=0.05)


def test_summarize_family_counts_wins_losses_nonsig():
    p_values = {"a": 0.001, "b": 0.001, "c": 0.5}
    _, results = bonferroni_correct(p_values, alpha=0.05)
    # 'a' favors focal (positive diff), 'b' favors alternative (negative diff)
    directions = {"a": 1, "b": -1, "c": 1}
    summary = summarize_family(results, directions)
    assert summary["wins"] == 1
    assert summary["losses"] == 1
    assert summary["nonsignificant"] == 1
    assert summary["detail"]["a"] == "win"
    assert summary["detail"]["b"] == "loss"
    assert summary["detail"]["c"] == "nonsignificant"


def test_summarize_family_requires_direction_for_every_row():
    p_values = {"a": 0.001}
    _, results = bonferroni_correct(p_values, alpha=0.05)
    with pytest.raises(ValueError):
        summarize_family(results, directions={})
