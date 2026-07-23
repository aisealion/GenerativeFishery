from typing import TypeVar

import anthropic
from pydantic import BaseModel, ValidationError

from genfishery.config.model_config import LLMCallType, ModelConfig
from genfishery.llm.client import LLMStructuredCallError

T = TypeVar("T", bound=BaseModel)

_TOOL_NAME = "respond"


class AnthropicLLMClient:
    def __init__(self, model_config: ModelConfig, api_key: str | None = None) -> None:
        self._model_config = model_config
        self._client = anthropic.AsyncAnthropic(api_key=api_key)

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

        response = await self._client.messages.create(
            model=params.model,
            max_tokens=params.max_tokens,
            temperature=params.temperature,
            system=system,
            messages=[{"role": "user", "content": prompt}],
            tools=[
                {
                    "name": _TOOL_NAME,
                    "description": f"Respond with {response_model.__name__}.",
                    "input_schema": schema,
                }
            ],
            tool_choice={"type": "tool", "name": _TOOL_NAME},
        )

        tool_use = next(
            (block for block in response.content if block.type == "tool_use"),
            None,
        )
        if tool_use is None:
            raise LLMStructuredCallError(
                f"{call_type}: model returned no tool_use block (stop_reason={response.stop_reason!r})"
            )

        try:
            return response_model.model_validate(tool_use.input)
        except ValidationError as exc:
            raise LLMStructuredCallError(
                f"{call_type}: tool_use input failed {response_model.__name__} validation: {exc}"
            ) from exc
