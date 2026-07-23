"""Tests for LLM_PROVIDER selection (litellm/ollama/anthropic)."""

import pytest

from genfishery.config.model_config import LLMCallType
from genfishery.llm.anthropic_client import AnthropicLLMClient
from genfishery.llm.openai_compatible_client import OpenAICompatibleLLMClient
from genfishery.llm.provider import build_llm_client, resolve_model_config


def test_resolve_model_config_litellm_covers_every_call_type_with_gpt5_4():
    config = resolve_model_config("litellm")
    for call_type in LLMCallType:
        assert config.for_call(call_type).model == "gpt-5.4"


def test_resolve_model_config_ollama_covers_every_call_type_with_local_model():
    config = resolve_model_config("ollama")
    for call_type in LLMCallType:
        assert config.for_call(call_type).model == "gpt-oss:20b"


def test_resolve_model_config_anthropic_still_tiers_haiku_and_sonnet():
    config = resolve_model_config("anthropic")
    assert config.for_call(LLMCallType.EFFORT_DECISION).model == "claude-haiku-4-5-20251001"
    assert config.for_call(LLMCallType.NORM_COMPILER).model == "claude-sonnet-5"


def test_resolve_model_config_unknown_provider_raises():
    with pytest.raises(ValueError, match="unknown LLM_PROVIDER"):
        resolve_model_config("not-a-real-provider")


def test_build_llm_client_litellm_requires_api_key(monkeypatch):
    monkeypatch.delenv("LITELLM_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="LITELLM_API_KEY"):
        build_llm_client("litellm", resolve_model_config("litellm"))


def test_build_llm_client_litellm_builds_openai_compatible_client(monkeypatch):
    monkeypatch.setenv("LITELLM_API_KEY", "sk-test")
    client = build_llm_client("litellm", resolve_model_config("litellm"))
    assert isinstance(client, OpenAICompatibleLLMClient)
    assert str(client._client.base_url).startswith("https://llm.uod.otago.ac.nz")


def test_build_llm_client_litellm_respects_base_url_override(monkeypatch):
    monkeypatch.setenv("LITELLM_API_KEY", "sk-test")
    monkeypatch.setenv("LITELLM_BASE_URL", "https://custom.example.com/v1")
    client = build_llm_client("litellm", resolve_model_config("litellm"))
    assert str(client._client.base_url).startswith("https://custom.example.com")


def test_build_llm_client_ollama_needs_no_api_key(monkeypatch):
    monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)
    client = build_llm_client("ollama", resolve_model_config("ollama"))
    assert isinstance(client, OpenAICompatibleLLMClient)
    assert str(client._client.base_url).startswith("http://localhost:11434")


def test_build_llm_client_anthropic_builds_anthropic_client(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    client = build_llm_client("anthropic", resolve_model_config("anthropic"))
    assert isinstance(client, AnthropicLLMClient)


def test_build_llm_client_unknown_provider_raises():
    with pytest.raises(ValueError, match="unknown LLM_PROVIDER"):
        build_llm_client("not-a-real-provider", resolve_model_config("anthropic"))
