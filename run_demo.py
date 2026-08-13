#!/usr/bin/env python3
"""
End-to-end demo: runs the full AFIP pipeline (Algorithm 1) across the
9-case sample dataset for three MockClient "backbones" with different
seeded error rates, computes paired-bootstrap + Bonferroni-corrected
significance, verifies the audit-log hash chain, and writes results/.

    python run_demo.py

*** Uses MockClient (synthetic, seeded data) — see afip/clients/mock_client.py.
*** To reproduce the paper's real claims, edit BACKBONES below to use
*** afip.clients.anthropic_client.AnthropicClient (and equivalents) with a
*** real, full-scale, expert-labelled dataset. See README.md.
"""
from __future__ import annotations
import json
from pathlib import Path

from afip.clients.mock_client import MockClient
from afip.evaluation.harness import load_cases, run_evaluation, write_results

REPO_ROOT = Path(__file__).parent


def main():
    cases = load_cases(REPO_ROOT / "data" / "sample_cases.jsonl")
    print(f"Loaded {len(cases)} demo cases from data/sample_cases.jsonl")

    # Three synthetic "backbones" with different, declared error profiles.
    # These numbers are seeded and known by construction — they exist to
    # prove the pipeline + stats code work, NOT as a claim about any real
    # model's accuracy.
    backbones = {
        "focal_low_error":  MockClient("focal_low_error",  seed=1, error_rate=0.05, schema_violation_rate=0.02, mean_latency=1.2),
        "alt_medium_error": MockClient("alt_medium_error", seed=2, error_rate=0.30, schema_violation_rate=0.05, mean_latency=1.6),
        "alt_high_error":   MockClient("alt_high_error",   seed=3, error_rate=0.55, schema_violation_rate=0.10, mean_latency=2.1),
    }

    audit_path = REPO_ROOT / "results" / "audit_log.jsonl"
    if audit_path.exists():
        audit_path.unlink()  # start each demo run from a clean log

    results = run_evaluation(
        cases=cases,
        backbones=backbones,
        focal_backbone="focal_low_error",
        audit_log_path=audit_path,
        n_resamples=10_000,
        seed=42,
    )

    write_results(results, out_dir=REPO_ROOT / "results")

    print("\n=== Per-backbone summary ===")
    for name, stats in results["per_backbone"].items():
        print(f"  {name:20s}  mean_score={stats['mean_score']:.3f}  "
              f"schema_fidelity={stats['schema_fidelity']:.3f}  "
              f"safety_block_rate={stats['safety_block_rate']:.3f}  "
              f"mean_latency={stats['mean_latency_seconds']:.3f}s  "
              f"median_latency={stats['median_latency_seconds']:.3f}s  "
              f"p90_latency={stats['p90_latency_seconds']:.3f}s")
        safety = stats["safety"]
        recall = safety["unsafe_detection_recall"]
        fpr = safety["benign_false_positive_rate"]
        recall_s = f"{recall:.3f}" if recall is not None else "n/a"
        fpr_s = f"{fpr:.3f}" if fpr is not None else "n/a"
        print(f"    safety: unsafe_detection_recall={recall_s}  "
              f"benign_false_positive_rate={fpr_s}  counts={safety['counts']}")

    print(f"\n=== Significance: {results['focal_backbone']} vs. each alternative "
          f"(Bonferroni over {len(results['significance'])} comparisons) ===")
    for row in results["significance"]:
        print(f"  vs {row['backbone']:20s} diff={row['observed_diff']:+.3f}  "
              f"p_raw={row['p_raw']:.5f}  alpha_corrected={row['bonferroni_alpha']:.6f}  "
              f"-> {row['outcome']}")

    print(f"\n=== Family summary === {results['family_summary']}")
    print(f"\nAudit log hash-chain verified: {results['audit_log_verified']}")
    print("\nWrote results/results.csv and results/summary.json")


if __name__ == "__main__":
    main()
