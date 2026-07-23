"""Provider selection: which LLM backend actually answers `structured_call`.

`LLM_PROVIDER` env var chooses between:
  - "litellm" (default): a self-hosted LiteLLM proxy, OpenAI-compatible,
    e.g. routing to GPT-5.4. Needs LITELLM_API_KEY; LITELLM_BASE_URL
    defaults to the project's proxy.
  - "ollama": a local Ollama server (also OpenAI-compatible at /v1), e.g.
    gpt-oss:20b. No API key needed.
  - "anthropic": the original spec-mandated Claude path.

Each provider has its own `configs/models_<provider>.yaml` -- model strings
are provider-specific (a Claude model id is meaningless to a LiteLLM/Ollama
endpoint and vice versa), so, unlike the single shared `models.yaml` the
Anthropic-only path used, every provider's YAML must cover every
`LLMCallType` explicitly: `load_model_config`'s fallback-to-defaults only
knows Anthropic model ids, so a call type left unspecified for a non-Anthropic
provider would silently try to invoke a Claude model string against that
provider's endpoint and fail.
"""

import os
from pathlib import Path

from genfishery.config.model_config import ModelConfig, load_model_config
from genfishery.llm.anthropic_client import AnthropicLLMClient
from genfishery.llm.client import LLMClient
from genfishery.llm.openai_compatible_client import OpenAICompatibleLLMClient

CONFIGS_DIR = Path(__file__).resolve().parents[3] / "configs"

DEFAULT_LITELLM_BASE_URL = "https://llm.uod.otago.ac.nz/v1"
DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434/v1"

_MODEL_CONFIG_FILENAMES = {
    "litellm": "models_litellm.yaml",
    "ollama": "models_ollama.yaml",
    "anthropic": "models.yaml",
}


def resolve_model_config(provider: str) -> ModelConfig:
    filename = _MODEL_CONFIG_FILENAMES.get(provider)
    if filename is None:
        raise ValueError(f"unknown LLM_PROVIDER: {provider!r}")
    return load_model_config(CONFIGS_DIR / filename)


def build_llm_client(provider: str, model_config: ModelConfig) -> LLMClient:
    if provider == "litellm":
        api_key = os.environ.get("LITELLM_API_KEY")
        if not api_key:
            raise RuntimeError("LITELLM_API_KEY must be set to use LLM_PROVIDER=litellm")
        base_url = os.environ.get("LITELLM_BASE_URL", DEFAULT_LITELLM_BASE_URL)
        return OpenAICompatibleLLMClient(model_config, base_url=base_url, api_key=api_key)

    if provider == "ollama":
        base_url = os.environ.get("OLLAMA_BASE_URL", DEFAULT_OLLAMA_BASE_URL)
        # Ollama's OpenAI-compatible endpoint ignores the API key but the
        # OpenAI SDK requires a non-empty string to construct the client.
        return OpenAICompatibleLLMClient(model_config, base_url=base_url, api_key="ollama")

    if provider == "anthropic":
        return AnthropicLLMClient(model_config)

    raise ValueError(f"unknown LLM_PROVIDER: {provider!r}")


def build_default_llm_client() -> tuple[LLMClient, ModelConfig]:
    """Reads `LLM_PROVIDER` (defaults to "litellm") and builds the matching
    client + its provider-scoped model config together, since one is
    meaningless without the other.
    """
    provider = os.environ.get("LLM_PROVIDER", "litellm").lower()
    model_config = resolve_model_config(provider)
    return build_llm_client(provider, model_config), model_config
