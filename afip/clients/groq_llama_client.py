"""
Llama 3.3 70B via Groq — free-tier backbone client.

Paper reference: same interface as afip.clients.anthropic_client.AnthropicClient
(Algorithm 1, Step 4), routed through Groq instead of a paid/self-hosted
endpoint. Groq's free tier (as of mid-2026) requires no credit card and
hosts Llama 3.3 70B directly with generous rate limits (~1,000 requests/day),
making it the most practical way to get real Llama 3.3 70B backbone results
for Table 6A / 7A / 7B without payment friction.

Note: the paper's Table 1 specifies "LLaMA 3 70B"; Groq's current free
offering is "llama-3.3-70b-versatile". Document the exact model string
actually used in Section 4.1 once you run this, since 3.0 vs 3.3 is a real
version difference a reviewer could ask about.

Requires: pip install groq
Auth: export GROQ_API_KEY=... (from console.groq.com, no card required to
create a free-tier key as of this writing — verify current terms yourself).
"""

from __future__ import annotations
import os
import time
import json

from .base import LLMClient, ToolUseResult

try:
    from groq import Groq
except ImportError:  # pragma: no cover
    Groq = None


class GroqLlamaClient(LLMClient):
    def __init__(self, model_id: str = "llama-3.3-70b-versatile", api_key: str | None = None):
        if Groq is None:
            raise ImportError(
                "The 'groq' package is required for GroqLlamaClient. "
                "Install with: pip install groq"
            )
        key = api_key or os.environ.get("GROQ_API_KEY")
        if not key:
            raise RuntimeError(
                "GROQ_API_KEY not set. Export it or pass api_key= explicitly. "
                "Get a free-tier key at https://console.groq.com (verify current "
                "card requirements yourself before relying on this)."
            )
        self.model_id = model_id
        self._client = Groq(api_key=key)

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
        # ground_truth / case_id intentionally unused: this is the real-API path.
        start = time.monotonic()

        # Groq's API is OpenAI-compatible; use function/tool calling in that shape.
        groq_tool = {
            "type": "function",
            "function": {
                "name": tool_schema["name"],
                "description": tool_schema.get("description", ""),
                "parameters": tool_schema["input_schema"],
            },
        }

        groq_messages = [{"role": "system", "content": system}] + messages

        response = self._client.chat.completions.create(
            model=self.model_id,
            messages=groq_messages,
            tools=[groq_tool],
            tool_choice={"type": "function", "function": {"name": tool_schema["name"]}},
            max_tokens=max_tokens,
            temperature=temperature,
        )
        latency = time.monotonic() - start

        tool_input = {}
        message = response.choices[0].message
        if message.tool_calls:
            raw_args = message.tool_calls[0].function.arguments
            try:
                tool_input = json.loads(raw_args)
            except (json.JSONDecodeError, TypeError):
                tool_input = {}

        return ToolUseResult(
            model_id=self.model_id,
            tool_name=tool_schema["name"],
            tool_input=tool_input,
            raw_response=response,
            latency_seconds=latency,
        )
