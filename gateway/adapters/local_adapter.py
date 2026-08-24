"""Local OpenAI-compatible inference server adapter."""
from .openai_adapter import OpenAIAdapter


class LocalAdapter(OpenAIAdapter):
    """Uses the standard OpenAI-compatible protocol for local inference servers."""
