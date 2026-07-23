"""LLMClient implementation for any OpenAI-compatible chat-completions
endpoint -- a self-hosted LiteLLM proxy (routing to e.g. GPT-5.4) or a local
Ollama server (both speak the same `/chat/completions` + tool-calling API),
selected via `LLM_PROVIDER` in `genfishery.llm.provider`.

Retries up to `_MAX_ATTEMPTS` times on a missing tool call or a schema
validation failure, feeding the error back to the model before trying again.
Local/smaller models (e.g. Ollama's gpt-oss, a reasoning model that can blow
through a tight max_tokens budget on chain-of-thought before ever emitting a
tool call, or return a number outside a JSON schema's min/max bounds) are
meaningfully less reliable at structured output than a frontier model, and
one bad response shouldn't crash the whole round. This still never
improvises a substitute answer (build spec §0/§2) -- it only gives the model
more chances to produce a schema-conforming one; the final attempt's failure
is raised exactly as before if every attempt fails.
"""

import json
from typing import TypeVar

import openai
from pydantic import BaseModel, ValidationError

from genfishery.config.model_config import LLMCallType, ModelConfig
from genfishery.llm.client import LLMStructuredCallError

T = TypeVar("T", bound=BaseModel)

_TOOL_NAME = "respond"
_MAX_ATTEMPTS = 3


class OpenAICompatibleLLMClient:
    def __init__(self, model_config: ModelConfig, *, base_url: str, api_key: str) -> None:
        self._model_config = model_config
        self._client = openai.AsyncOpenAI(base_url=base_url, api_key=api_key)

    async def structured_call(
        self,
        *,
        call_type: LLMCallType,
        system: str,
        prompt: str,
        response_model: type[T],
    ) -> T:
        params = self._model_config.for_call(call_type)
        schema = response_model.model_json_schema()
        schema.pop("title", None)
        tools = [
            {
                "type": "function",
                "function": {
                    "name": _TOOL_NAME,
                    "description": f"Respond with {response_model.__name__}.",
                    "parameters": schema,
                },
            }
        ]
        messages: list[dict] = [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ]

        last_error = LLMStructuredCallError(f"{call_type}: no attempts succeeded")
        for _ in range(_MAX_ATTEMPTS):
            response = await self._client.chat.completions.create(
                model=params.model,
                max_tokens=params.max_tokens,
                temperature=params.temperature,
                messages=messages,
                tools=tools,
                tool_choice={"type": "function", "function": {"name": _TOOL_NAME}},
            )

            message = response.choices[0].message
            tool_calls = message.tool_calls or []
            if not tool_calls:
                last_error = LLMStructuredCallError(
                    f"{call_type}: model returned no tool call "
                    f"(finish_reason={response.choices[0].finish_reason!r})"
                )
                messages.append(
                    {"role": "user", "content": "You must respond by calling the tool now, with no other text."}
                )
                continue

            try:
                arguments = json.loads(tool_calls[0].function.arguments)
                return response_model.model_validate(arguments)
            except (json.JSONDecodeError, ValidationError) as exc:
                last_error = LLMStructuredCallError(
                    f"{call_type}: tool call arguments failed {response_model.__name__} validation: {exc}"
                )
                messages.append(
                    {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "id": tool_calls[0].id,
                                "type": "function",
                                "function": {
                                    "name": _TOOL_NAME,
                                    "arguments": tool_calls[0].function.arguments,
                                },
                            }
                        ],
                    }
                )
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_calls[0].id,
                        "content": f"That call was invalid: {exc}. Call the tool again with corrected arguments.",
                    }
                )

        raise last_error
