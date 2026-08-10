"""
LLMClient interface.

Paper reference: Section 5.1, "Every Claude inference call across all five
algorithms uses the same Python Anthropic SDK pattern... Variable names,
parameter order and error handling are identical across modules."

This abstract base class is what makes the six-backbone comparison in
Table 6A / Table 7 actually swappable in code: Algorithm 1
(afip.algorithms.master_orchestration.run) is written against this
interface only, so running the same evaluation harness against
AnthropicClient, an OpenAI-compatible client, or MockClient requires no
change to orchestration logic — exactly the "keeping AFIP wrapper,
MODULE_SCHEMA definitions... identical" condition described for the
six-backbone experiment (Section 6, Table 6A).
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class ToolUseResult:
    """Normalized result of a single model call, independent of provider."""
    model_id: str
    tool_name: str
    tool_input: dict[str, Any]
    raw_response: Any
    latency_seconds: float


class LLMClient(ABC):
    """Common interface every backbone client (real or mock) must implement."""

    #: Identifier reported in audit logs and results tables (e.g. "claude-sonnet-4-6").
    model_id: str

    @abstractmethod
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
        """
        Issue one constrained (tool-use / function-calling) inference call.

        Implementations MUST force the model to return JSON conforming to
        `tool_schema["input_schema"]` via each provider's native structured-
        output mechanism (Anthropic tool_choice, OpenAI function calling,
        etc.), mirroring Algorithm 1 Step 4 in the paper.

        `ground_truth` / `case_id` are optional and exist only so that
        MockClient (offline demo / unit tests) can synthesize a plausible,
        seeded, imperfect prediction without a real model call. Real
        clients (e.g. AnthropicClient) accept and ignore these arguments —
        they are never sent to the provider API.
        """
        raise NotImplementedError
