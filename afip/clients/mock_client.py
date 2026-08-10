"""
MockClient — a synthetic, seeded stand-in for a real LLM backbone.

*** THIS CLIENT PRODUCES SIMULATED DATA. IT DOES NOT CALL ANY MODEL. ***

Why this exists
----------------
The V1/V2 manuscript reported precise benchmark numbers (Tables 6, 6A, 7)
with no runnable protocol behind them — the single largest reproducibility
defect identified in review. This client exists so that:

  1. The full AFIP pipeline (Algorithm 1: safety gate -> retrieval ->
     routing -> constrained inference -> validation/retry -> formatting ->
     audit log) is demonstrably executable end-to-end, right now, with no
     API key and no network access, and
  2. `afip.evaluation.harness` can be smoke-tested and its statistics
     module (bootstrap + Bonferroni correction) can be exercised against
     data with a *known, seeded* ground-truth error rate, so the stats
     code can be verified against a known-correct answer.

It intentionally injects configurable, seeded error/hallucination and
schema-violation rates per "persona" so that the difference between two
mock backbones is a controlled, known quantity — the opposite of the
paper's original problem, where six real models' numbers appeared with no
declared data-generating process.

To reproduce the paper's actual claims, swap MockClient for
afip.clients.anthropic_client.AnthropicClient (and equivalent clients for
the other five backbones) and run against a real, IRB/ethics-appropriate,
expert-labelled dataset. MockClient output must never be reported as a
benchmark result in the paper — only used to validate that the code runs.
"""

from __future__ import annotations
import hashlib
import random
import time

from .base import LLMClient, ToolUseResult


class MockClient(LLMClient):
    def __init__(
        self,
        model_id: str,
        seed: int = 42,
        error_rate: float = 0.10,
        schema_violation_rate: float = 0.02,
        mean_latency: float = 1.5,
    ):
        """
        error_rate: probability that a numeric/categorical field is
            perturbed away from ground truth (simulates a "hallucination"
            or wrong extraction for benchmark purposes).
        schema_violation_rate: probability the first attempt returns a
            structurally invalid payload, forcing Algorithm 1's retry loop
            (Step 4: "Repeat up to 3 times") to actually execute.
        """
        self.model_id = model_id
        self.seed = seed
        self.error_rate = error_rate
        self.schema_violation_rate = schema_violation_rate
        self.mean_latency = mean_latency
        self._attempt_counts: dict[str, int] = {}

    def _rng_for(self, case_id: str, attempt: int) -> random.Random:
        # Deterministic per (model, case, attempt) seed -> fully reproducible.
        key = f"{self.model_id}|{case_id}|{attempt}|{self.seed}"
        digest = hashlib.sha256(key.encode()).hexdigest()
        return random.Random(int(digest[:16], 16))

    def create_tool_call(
        self,
        system: str,
        tool_schema: dict,
        messages: list[dict],
        temperature: float = 0.0,
        max_tokens: int = 1024,
        ground_truth: dict | None = None,
        case_id: str | None = None,
    ) -> ToolUseResult:
        case_id = case_id or "unkeyed"
        attempt = self._attempt_counts.get(case_id, 0)
        self._attempt_counts[case_id] = attempt + 1
        rng = self._rng_for(case_id, attempt)

        start = time.monotonic()
        latency = max(0.05, rng.gauss(self.mean_latency, self.mean_latency * 0.2))

        if rng.random() < self.schema_violation_rate:
            # Simulate a malformed / schema-invalid first attempt.
            payload = {"__malformed__": True}
        else:
            payload = self._synthesize(tool_schema, ground_truth or {}, rng)

        return ToolUseResult(
            model_id=self.model_id,
            tool_name=tool_schema["name"],
            tool_input=payload,
            raw_response={"mock": True, "attempt": attempt},
            latency_seconds=latency,
        )

    def _synthesize(self, tool_schema: dict, ground_truth: dict, rng: random.Random) -> dict:
        props = tool_schema["input_schema"]["properties"]
        out = {}
        for key, spec in props.items():
            if key in ground_truth:
                out[key] = self._perturb(ground_truth[key], spec, rng)
            else:
                out[key] = self._default_for(spec, rng)
        return out

    def _perturb(self, true_value, spec: dict, rng: random.Random):
        if rng.random() >= self.error_rate:
            return true_value  # correct prediction
        t = spec.get("type")
        if t == "number":
            noise = rng.uniform(0.15, 0.6) * (1 if rng.random() < 0.5 else -1)
            return round(float(true_value) * (1 + noise), 4)
        if t == "boolean":
            return not true_value
        if t == "string" and "enum" in spec:
            others = [v for v in spec["enum"] if v != true_value]
            return rng.choice(others) if others else true_value
        if t == "array":
            return []
        return true_value

    def _default_for(self, spec: dict, rng: random.Random):
        t = spec.get("type")
        if t == "number":
            lo = spec.get("minimum", 0.0)
            hi = spec.get("maximum", 1.0)
            return round(rng.uniform(lo, hi), 4)
        if t == "boolean":
            return rng.random() < 0.5
        if t == "string" and "enum" in spec:
            return rng.choice(spec["enum"])
        if t == "string":
            return "mock text"
        if t == "array":
            return []
        return None
