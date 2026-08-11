from afip.clients.mock_client import MockClient
from afip.clients.base import LLMClient


def test_vertex_client_module_importable_without_credentials():
    """
    The vertex_client module itself should always be importable (the
    AnthropicVertex import is wrapped in try/except); actually constructing
    VertexClaudeClient without the vertex extra installed, or without GCP
    credentials, should raise a clear error rather than crash silently.
    """
    from afip.clients import vertex_client
    assert hasattr(vertex_client, "VertexClaudeClient")


def test_mock_client_satisfies_llmclient_interface():
    client = MockClient("test-model")
    assert isinstance(client, LLMClient)
    assert client.model_id == "test-model"
