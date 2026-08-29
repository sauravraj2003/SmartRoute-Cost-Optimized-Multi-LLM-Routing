from gateway.adapters.anthropic_adapter import AnthropicAdapter
from gateway.adapters.local_adapter import LocalAdapter
from gateway.adapters.openai_adapter import OpenAIAdapter
from gateway.api import _adapter


def test_adapter_factory_maps_supported_providers():
    assert isinstance(_adapter({"id": "x", "provider": "openai"}), OpenAIAdapter)
    assert isinstance(_adapter({"id": "x", "provider": "groq"}), OpenAIAdapter)
    assert isinstance(_adapter({"id": "x", "provider": "local"}), LocalAdapter)
    assert isinstance(_adapter({"id": "x", "provider": "anthropic"}), AnthropicAdapter)
