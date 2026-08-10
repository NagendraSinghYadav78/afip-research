"""
Statistical methodology used for Tables 6A / 7 comparisons.

Paper reference: Section 6/7, "P-values from paired bootstrap (10,000
resamples) of Claude versus the next best alternative... Bonferroni
corrected over the 14 metric rows at alpha = 0.05/14."

This module is the actual, runnable implementation of that sentence — the
single biggest reproducibility gap flagged in review (both rounds). Given
two paired per-case score arrays (same test cases, two backbones), it:

  1. computes an observed paired mean difference,
  2. runs a paired bootstrap hypothesis test (resampling case indices, not
     the two conditions independently, to respect the paired design),
  3. returns a two-sided p-value,
  4. applies a Bonferroni correction across a declared family size.

All functions are pure and unit-tested (tests/test_stats.py) against cases
with a known answer (identical distributions -> p not significant;
non-overlapping distributions -> p significant), which is the concrete,
checkable thing "reproducible statistics" needs to mean.
"""

from __future__ import annotations
from dataclasses import dataclass

import numpy as np


@dataclass
class BootstrapResult:
    observed_diff: float
    p_value: float
    ci_low: float
    ci_high: float
    n_resamples: int


def paired_bootstrap_test(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    n_resamples: int = 10_000,
    seed: int = 42,
    ci: float = 0.95,
) -> BootstrapResult:
    """
    Paired bootstrap test for H0: mean(scores_a - scores_b) == 0.

    scores_a, scores_b: equal-length arrays, one value per matched test
    case (e.g. per-case correctness, or a per-case error magnitude), for
    the SAME cases evaluated under two backbones/configurations.

    Returns the observed mean difference, a two-sided bootstrap p-value,
    and a percentile confidence interval for the difference — exactly the
    quantities the paper's tables report per row.
    """
    a = np.asarray(scores_a, dtype=float)
    b = np.asarray(scores_b, dtype=float)
    if a.shape != b.shape:
        raise ValueError("scores_a and scores_b must be the same shape (paired design)")
    if a.ndim != 1 or a.size == 0:
        raise ValueError("scores must be non-empty 1-D arrays")

    n = a.size
    diff = a - b
    observed = float(diff.mean())

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_resamples, n))
    resampled_diffs = diff[idx].mean(axis=1)

    # Two-sided p-value: test whether 0 is an extreme value of the
    # bootstrap distribution of the difference, centered appropriately.
    centered = resampled_diffs - resampled_diffs.mean() + 0  # null-centered at 0
    p_value = float(np.mean(np.abs(centered) >= abs(observed)))
    # Guard against p=0 from finite resampling.
    p_value = max(p_value, 1.0 / (n_resamples + 1))

    alpha_tail = (1 - ci) / 2
    ci_low, ci_high = np.quantile(resampled_diffs, [alpha_tail, 1 - alpha_tail])

    return BootstrapResult(
        observed_diff=observed,
        p_value=p_value,
        ci_low=float(ci_low),
        ci_high=float(ci_high),
        n_resamples=n_resamples,
    )


@dataclass
class CorrectedResult:
    label: str
    p_raw: float
    p_threshold: float
    significant: bool


def bonferroni_correct(
    p_values: dict[str, float],
    alpha: float = 0.05,
) -> tuple[float, list[CorrectedResult]]:
    """
    Family-wise Bonferroni correction.

    p_values: mapping of {row_label: raw p-value}. The family size is
    len(p_values) — callers MUST pass exactly the set of tests that will
    be jointly reported (this is what the V2 review flagged: rows with
    n/a or untested metrics must not be silently counted into, or left
    out of, the family without saying so).

    Returns (corrected_alpha, per-row results).
    """
    m = len(p_values)
    if m == 0:
        raise ValueError("p_values must be non-empty")
    corrected_alpha = alpha / m
    results = [
        CorrectedResult(label=k, p_raw=v, p_threshold=corrected_alpha, significant=v < corrected_alpha)
        for k, v in p_values.items()
    ]
    return corrected_alpha, results


def summarize_family(results: list[CorrectedResult], directions: dict[str, int]) -> dict:
    """
    Reduce a family of corrected per-row results to the win/loss/ns counts
    the paper reports in prose (e.g. "9 significant wins, 2 significant
    losses, 3 non-significant differences").

    directions: {row_label: +1 if a positive observed difference favors
    the focal backbone, -1 if a negative difference favors it (e.g.
    hallucination rate, where lower is better)}. Must contain every label
    in `results`.
    """
    wins = losses = nonsig = 0
    detail = {}
    for r in results:
        d = directions.get(r.label)
        if d is None:
            raise ValueError(f"missing direction for '{r.label}'")
        if not r.significant:
            nonsig += 1
            outcome = "nonsignificant"
        elif d > 0:
            wins += 1
            outcome = "win"
        else:
            losses += 1
            outcome = "loss"
        detail[r.label] = outcome
    return {"wins": wins, "losses": losses, "nonsignificant": nonsig, "detail": detail}
