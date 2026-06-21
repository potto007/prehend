from typing import Any

from dotenv import load_dotenv

from prehend.clients.base_lm import BaseLM
from prehend.core.types import ClientBackend

load_dotenv()


def get_client(
    backend: ClientBackend,
    backend_kwargs: dict[str, Any],
) -> BaseLM:
    """
    Routes a specific backend and the args (as a dict) to the appropriate client if supported.
    Currently supported backends: ['openai']
    """
    if backend == "openai":
        from prehend.clients.openai import OpenAIClient

        return OpenAIClient(**backend_kwargs)
    elif backend == "vllm":
        from prehend.clients.openai import OpenAIClient

        assert "base_url" in backend_kwargs, (
            "base_url is required to be set to local vLLM server address for vLLM"
        )
        return OpenAIClient(**backend_kwargs)
    elif backend == "portkey":
        from prehend.clients.portkey import PortkeyClient

        return PortkeyClient(**backend_kwargs)
    elif backend == "openrouter":
        from prehend.clients.openai import OpenAIClient

        backend_kwargs.setdefault("base_url", "https://openrouter.ai/api/v1")
        return OpenAIClient(**backend_kwargs)
    elif backend == "vercel":
        from prehend.clients.openai import OpenAIClient

        backend_kwargs.setdefault("base_url", "https://ai-gateway.vercel.sh/v1")
        return OpenAIClient(**backend_kwargs)
    elif backend == "anthropic":
        from prehend.clients.anthropic import AnthropicClient

        return AnthropicClient(**backend_kwargs)
    elif backend == "gemini":
        from prehend.clients.gemini import GeminiClient

        return GeminiClient(**backend_kwargs)
    elif backend == "azure_openai":
        from prehend.clients.azure_openai import AzureOpenAIClient

        return AzureOpenAIClient(**backend_kwargs)
    else:
        raise ValueError(
            f"Unknown backend: {backend}. Supported backends: ['openai', 'vllm', 'portkey', 'openrouter', 'anthropic', 'azure_openai', 'gemini', 'vercel']"
        )
