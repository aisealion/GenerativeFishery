"""Tests for OpenAICompatibleLLMClient (LiteLLM proxy / local Ollama path).

Mocks the openai SDK's chat.completions.create -- there's no live proxy or
Ollama server available in CI, so this verifies the request/response
plumbing (tool-forcing shape, argument parsing, error handling) rather than
hitting a real endpoint.
"""

import json
from unittest.mock import AsyncMock, MagicMock

import httpx
import openai
import pytest
from pydantic import BaseModel, Field

from genfishery.config.model_config import LLMCallType, ModelConfig
from genfishery.llm.client import LLMStructuredCallError
from genfishery.llm.openai_compatible_client import OpenAICompatibleLLMClient


class Decision(BaseModel):
    effort: float = Field(ge=0.0, le=1.0)


def make_client() -> OpenAICompatibleLLMClient:
    return OpenAICompatibleLLMClient(
        ModelConfig.default(), base_url="http://example.invalid/v1", api_key="test-key"
    )


def _mock_response(tool_arguments: dict | None, finish_reason: str = "tool_calls"):
    response = MagicMock()
    choice = MagicMock()
    choice.finish_reason = finish_reason
    if tool_arguments is None:
        choice.message.tool_calls = []
    else:
        tool_call = MagicMock()
        tool_call.function.arguments = json.dumps(tool_arguments)
        choice.message.tool_calls = [tool_call]
    response.choices = [choice]
    return response


async def test_structured_call_sends_forced_tool_choice_and_schema():
    client = make_client()
    client._client.chat.completions.create = AsyncMock(return_value=_mock_response({"effort": 0.4}))

    result = await client.structured_call(
        call_type=LLMCallType.EFFORT_DECISION, system="sys", prompt="p", response_model=Decision
    )

    assert result.effort == 0.4
    call_kwargs = client._client.chat.completions.create.call_args.kwargs
    assert call_kwargs["messages"] == [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "p"},
    ]
    assert call_kwargs["tool_choice"] == {"type": "function", "function": {"name": "respond"}}
    assert call_kwargs["tools"][0]["function"]["name"] == "respond"
    assert call_kwargs["tools"][0]["function"]["parameters"]["properties"]["effort"]["type"] == "number"
    params = ModelConfig.default().for_call(LLMCallType.EFFORT_DECISION)
    assert call_kwargs["model"] == params.model
    assert call_kwargs["max_tokens"] == params.max_tokens


async def test_structured_call_raises_when_no_tool_call_returned():
    client = make_client()
    client._client.chat.completions.create = AsyncMock(
        return_value=_mock_response(None, finish_reason="stop")
    )

    with pytest.raises(LLMStructuredCallError, match="no tool call"):
        await client.structured_call(
            call_type=LLMCallType.EFFORT_DECISION, system="sys", prompt="p", response_model=Decision
        )


async def test_structured_call_raises_on_schema_validation_failure():
    client = make_client()
    client._client.chat.completions.create = AsyncMock(
        return_value=_mock_response({"effort": "not a number"})
    )

    with pytest.raises(LLMStructuredCallError, match="failed .* validation"):
        await client.structured_call(
            call_type=LLMCallType.EFFORT_DECISION, system="sys", prompt="p", response_model=Decision
        )


async def test_structured_call_raises_on_malformed_json_arguments():
    client = make_client()
    response = MagicMock()
    tool_call = MagicMock()
    tool_call.function.arguments = "{not valid json"
    response.choices = [MagicMock(message=MagicMock(tool_calls=[tool_call]))]
    client._client.chat.completions.create = AsyncMock(return_value=response)

    with pytest.raises(LLMStructuredCallError, match="failed .* validation"):
        await client.structured_call(
            call_type=LLMCallType.EFFORT_DECISION, system="sys", prompt="p", response_model=Decision
        )


async def test_structured_call_retries_after_validation_failure_then_succeeds():
    client = make_client()
    client._client.chat.completions.create = AsyncMock(
        side_effect=[
            _mock_response({"effort": 5}),  # out of range -- rejected
            _mock_response({"effort": 0.4}),  # corrected on retry
        ]
    )

    result = await client.structured_call(
        call_type=LLMCallType.EFFORT_DECISION, system="sys", prompt="p", response_model=Decision
    )

    assert result.effort == 0.4
    assert client._client.chat.completions.create.await_count == 2
    # The retry's request includes the failed tool call and an error message
    # fed back, so the model sees what went wrong.
    second_call_messages = client._client.chat.completions.create.await_args_list[1].kwargs["messages"]
    assert second_call_messages[-2]["role"] == "assistant"
    assert second_call_messages[-1]["role"] == "tool"
    assert "invalid" in second_call_messages[-1]["content"]


async def test_structured_call_retries_after_no_tool_call_then_succeeds():
    client = make_client()
    client._client.chat.completions.create = AsyncMock(
        side_effect=[
            _mock_response(None, finish_reason="length"),
            _mock_response({"effort": 0.7}),
        ]
    )

    result = await client.structured_call(
        call_type=LLMCallType.EFFORT_DECISION, system="sys", prompt="p", response_model=Decision
    )

    assert result.effort == 0.7
    assert client._client.chat.completions.create.await_count == 2


def _internal_server_error(message: str) -> openai.InternalServerError:
    response = httpx.Response(status_code=500, request=httpx.Request("POST", "http://example.invalid/v1"))
    return openai.InternalServerError(message, response=response, body=None)


async def test_structured_call_retries_after_provider_5xx_then_succeeds():
    """Confirmed failure mode with Ollama + gpt-oss: a too-long answer cuts
    the tool call's JSON off mid-string, and Ollama fails constructing it
    server-side with a 500 ("unexpected end of JSON input") -- raised by the
    openai SDK as InternalServerError, never reaching the tool-call/JSON
    checks below, so it needs its own retry path.
    """
    client = make_client()
    client._client.chat.completions.create = AsyncMock(
        side_effect=[
            _internal_server_error("error parsing tool call: unexpected end of JSON input"),
            _mock_response({"effort": 0.6}),
        ]
    )

    result = await client.structured_call(
        call_type=LLMCallType.EFFORT_DECISION, system="sys", prompt="p", response_model=Decision
    )

    assert result.effort == 0.6
    assert client._client.chat.completions.create.await_count == 2
    second_call_messages = client._client.chat.completions.create.await_args_list[1].kwargs["messages"]
    assert second_call_messages[-1]["role"] == "user"
    assert "shorter" in second_call_messages[-1]["content"]


async def test_structured_call_gives_up_after_max_attempts():
    client = make_client()
    client._client.chat.completions.create = AsyncMock(return_value=_mock_response({"effort": 5}))

    with pytest.raises(LLMStructuredCallError, match="failed .* validation"):
        await client.structured_call(
            call_type=LLMCallType.EFFORT_DECISION, system="sys", prompt="p", response_model=Decision
        )

    assert client._client.chat.completions.create.await_count == 3
