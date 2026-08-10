"""
Real Anthropic API backbone client.

Paper reference: Algorithm 1, Step 4 ("Constitution-bounded LLM inference"):

    r = Claude.MessagesCreate(
        model = 'claude-sonnet-4-6',
        system = FINANCIAL_CONSTITUTION_PROMPT,
        tools = [MODULE_SCHEMA[m]],
        tool_choice = { type: 'tool', name: MODULE_SCHEMA[m].name },
        messages = [{ role: 'user', content: p }]
    )

This is that call, implemented against the real `anthropic` Python SDK.
Requires ANTHROPIC_API_KEY in the environment. Not used by the bundled
offline demo (see afip.clients.mock_client) but this is the client that
must be used to regenerate the paper's actual benchmark numbers.
"""

from __future__ import annotations
import os
import time

from .base import LLMClient, ToolUseResult

try:
    import anthropic
except ImportError:  # pragma: no cover
    anthropic = None


class AnthropicClient(LLMClient):
    def __init__(self, model_id: str = "claude-sonnet-4-6", api_key: str | None = None):
        if anthropic is None:
            raise ImportError(
                "The 'anthropic' package is required for AnthropicClient. "
                "Install with: pip install anthropic"
            )
        key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY not set. Export it or pass api_key= explicitly. "
                "This client makes real, billed API calls."
            )
        self.model_id = model_id
        self._client = anthropic.Anthropic(api_key=key)

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
        # ground_truth / case_id are intentionally unused here: this is the
        # real-API path and must never see evaluation labels.
        start = time.monotonic()
        response = self._client.messages.create(
            model=self.model_id,
            system=system,
            tools=[tool_schema],
            tool_choice={"type": "tool", "name": tool_schema["name"]},
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        latency = time.monotonic() - start

        tool_input = {}
        for block in response.content:
            if getattr(block, "type", None) == "tool_use":
                tool_input = block.input
                break

        return ToolUseResult(
            model_id=self.model_id,
            tool_name=tool_schema["name"],
            tool_input=tool_input,
            raw_response=response,
            latency_seconds=latency,
        )
