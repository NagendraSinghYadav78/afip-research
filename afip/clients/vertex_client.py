"""
Claude via Google Cloud Vertex AI — alternative backbone client.

Paper reference: same as afip.clients.anthropic_client.AnthropicClient
(Algorithm 1, Step 4), but routed through Google Cloud's Vertex AI instead
of Anthropic's direct API. Useful when direct Anthropic billing isn't
available (e.g. certain card/region restrictions) — same model, same
underlying `anthropic` Python SDK, different auth/billing path.

Requires: pip install "anthropic[vertex]"
Requires: a Google Cloud project with Vertex AI enabled and the Claude
model enabled in Model Garden (see project README / setup notes).
Auth: in Colab, run `from google.colab import auth; auth.authenticate_user()`
before constructing this client. Outside Colab, use Application Default
Credentials (`gcloud auth application-default login`).
"""

from __future__ import annotations
import time

from .base import LLMClient, ToolUseResult

try:
    from anthropic import AnthropicVertex
except ImportError:  # pragma: no cover
    AnthropicVertex = None


class VertexClaudeClient(LLMClient):
    def __init__(self, project_id: str, region: str, model_id: str = "claude-sonnet-4-6"):
        if AnthropicVertex is None:
            raise ImportError(
                "The 'anthropic' package with Vertex support is required. "
                "Install with: pip install \"anthropic[vertex]\""
            )
        self.model_id = model_id
        self._client = AnthropicVertex(project_id=project_id, region=region)

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
