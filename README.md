# AFIP Research Companion Code

Reproducible-research companion to the manuscript *"Constitutional AI
Aligned Financial Decision Making: A Cross Benchmark Empirical Evaluation
of Frontier LLMs and the AFIP Production Architecture."*

This repository exists to close the single largest reviewer objection
raised against the manuscript across two review rounds: **the paper
reported precise experimental numbers (Tables 6, 6A, 7) with no runnable
protocol behind them.** Every module here is a real, tested implementation
of a specific piece of that protocol, annotated with the exact paper
section it corresponds to.

## Status: honest scope

**What is real and runs today:**
- The full AFIP Algorithm 1 pipeline (safety gate → retrieval → routing →
  schema-constrained inference with retry → validation → formatting →
  hash-chained audit log), executable end to end with no API key.
- The statistical methodology the paper describes in prose — paired
  bootstrap significance testing and Bonferroni correction — as actual,
  unit-tested code (`afip/evaluation/stats.py`), not just a sentence.
- A small (9-case), original, hand-authored demo dataset spanning all four
  modules, used to prove the pipeline and statistics are wired correctly.
- 31 passing unit/integration tests (`pytest tests/`).

**What this is *not*, and should not be mistaken for:**
- This does **not** reproduce the paper's n=1,020 / 14-benchmark / 50,400-
  evaluation results. `run_demo.py` uses `MockClient`, a seeded synthetic
  stand-in (see `afip/clients/mock_client.py`) — it proves the *code*
  works, not that any real model achieves any particular score.
- Regenerating the paper's actual claims requires: (a) real API clients
  for all six backbones (only `AnthropicClient` is implemented here — the
  other five are one small subclass each, following the same
  `LLMClient` interface), (b) the full labelled evaluation dataset with a
  documented expert-annotation protocol, and (c) running at the scale the
  paper claims. None of that is fabricated here.

## Paper → code map

| Paper element | Code |
|---|---|
| Section 5.1, "MODULE_SCHEMA directory... enforce structured output" | `afip/schemas/module_schemas.py` |
| Section 5.1, "same Python Anthropic SDK pattern... identical across modules" | `afip/clients/base.py` (`LLMClient` interface) |
| Algorithm 1, Step 4 (`Claude.MessagesCreate(...)`) | `afip/clients/anthropic_client.py` |
| Algorithm 1, all 7 steps | `afip/algorithms/master_orchestration.py::run` |
| "DistilBERT safety classifier" (Limitations: unvalidated) | `default_safety_classifier` — explicit, transparent keyword screen, not a claimed DistilBERT model |
| "FAISS.kNN" retrieval | `default_retriever` — lexical-overlap stand-in, swappable |
| "SEC Rule 17a-4 ... immutable audit logging" | `AuditLog` — real hash-chained append-only log with `.verify()`, labelled as "record-retention-oriented," not a compliance certification |
| Section 6, "P-values from paired bootstrap (10,000 resamples)... Bonferroni corrected... α = 0.05/14" | `afip/evaluation/stats.py::paired_bootstrap_test`, `bonferroni_correct` |
| Table 6A / Table 7 win/loss/nonsignificant counts | `afip/evaluation/stats.py::summarize_family` |
| Section 6, "keeping AFIP wrapper... identical" across six backbones | `afip/evaluation/harness.py::run_evaluation` (backbone-agnostic by construction) |
| n=1,020 test suite | `data/sample_cases.jsonl` — **9-case original demo subset only**, see Status above |
| Section 4.6 "Metrics" (field-level correctness definitions) | `afip/evaluation/scoring.py::score_case` |

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest tests/ -v          # 31 tests, no API key required
python run_demo.py        # runs the full pipeline on synthetic backbones
```

`run_demo.py` prints per-backbone scores, schema fidelity, latency, a
Bonferroni-corrected paired-bootstrap significance table, and audit-log
verification, then writes `results/results.csv` and `results/summary.json`
— the same shape as the paper's Table 6A / Table 7.

## Running against a real model

```python
import os
from afip.clients.anthropic_client import AnthropicClient

os.environ["ANTHROPIC_API_KEY"] = "sk-..."
client = AnthropicClient(model_id="claude-sonnet-4-6")
```

Pass this (and equivalent clients you write for GPT-4o, Gemini, LLaMA,
Mistral, Grok — each is a ~40-line subclass of `LLMClient`) into
`afip.evaluation.harness.run_evaluation` in place of `MockClient` instances.
**This will make real, billed API calls.** Before running at the paper's
claimed scale, first write the full experimental-methodology subsection
(exact model IDs/dates, decoding parameters, dataset construction, expert
annotation protocol with inter-rater reliability) — the code will happily
run against whatever dataset you point it at, but a dataset with an
undocumented labelling process is still not a defensible scientific claim,
no matter how the code that scores it is written.

## Repository layout

```
afip/
  algorithms/master_orchestration.py   # Algorithm 1
  clients/base.py                      # LLMClient interface
  clients/anthropic_client.py          # real backbone
  clients/mock_client.py               # synthetic backbone (demo/tests only)
  schemas/module_schemas.py            # MODULE_SCHEMA (M1-M4)
  evaluation/scoring.py                # per-case correctness criteria
  evaluation/stats.py                  # paired bootstrap + Bonferroni
  evaluation/harness.py                # end-to-end evaluation runner
data/sample_cases.jsonl                # 9-case original demo dataset
tests/                                 # 31 pytest tests
run_demo.py                            # end-to-end demo entry point
results/                               # written by run_demo.py (gitignored except .gitkeep)
```

## License

MIT — see LICENSE.
