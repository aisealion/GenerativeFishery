"""LLMClient implementation for any OpenAI-compatible chat-completions
endpoint -- a self-hosted LiteLLM proxy (routing to e.g. GPT-5.4) or a local
Ollama server (both speak the same `/chat/completions` + tool-calling API),
selected via `LLM_PROVIDER` in `genfishery.llm.provider`.

Retries up to `_MAX_ATTEMPTS` times on a missing tool call, a schema
validation failure, a 5xx from the provider itself, or a request timeout,
feeding the error back to the model (where that's meaningful) before trying
again. Local/smaller models (e.g. Ollama's gpt-oss, a reasoning model that
can blow through a tight max_tokens budget on chain-of-thought before ever
emitting a tool call, or return a number outside a JSON schema's min/max
bounds) are meaningfully less reliable at structured output than a frontier
model, and one bad response shouldn't crash the whole round. The 5xx case is
a distinct failure mode confirmed in practice with Ollama + gpt-oss: a
sufficiently long answer runs out of max_tokens mid-way through the tool
call's JSON string, and Ollama's own tool-call construction fails
server-side with a 500 ("unexpected end of JSON input") rather than
returning a normal completion with a missing/malformed tool call -- so it
never reaches this client's own tool-call/JSON checks below at all, and
needs its own retry path. Timeouts are a separate failure mode, also
confirmed in practice on a shared HPC GPU node: a large local model's very
first inference call can take far longer than a warm one while Ollama loads
it into memory, easily exceeding a normal request timeout -- `_TIMEOUT_SECONDS`
is deliberately generous (overridable via `LLM_REQUEST_TIMEOUT_SECONDS`) to
absorb that, and a timeout is retried rather than crashing the whole round
on what's often a one-off cold-start cost. This still never improvises a
substitute answer (build spec §0/§2) -- it only gives the model more chances
to produce a schema-conforming one; the final attempt's failure is raised
exactly as before if every attempt fails.
"""

import json
import os
from typing import TypeVar

import openai
from pydantic import BaseModel, ValidationError

from genfishery.config.model_config import LLMCallType, ModelConfig
from genfishery.llm.client import LLMStructuredCallError

T = TypeVar("T", bound=BaseModel)

_TOOL_NAME = "respond"
_MAX_ATTEMPTS = 3
# The openai SDK's own default (600s total, 5s connect) has been observed too
# short for a large local model's cold first inference call on a shared HPC
# GPU node -- see the module docstring. 20 minutes gives real headroom while
# still failing well within any reasonable Slurm wall-time budget.
_TIMEOUT_SECONDS = float(os.environ.get("LLM_REQUEST_TIMEOUT_SECONDS", 1200.0))


class OpenAICompatibleLLMClient:
    def __init__(self, model_config: ModelConfig, *, base_url: str, api_key: str) -> None:
        self._model_config = model_config
        self._client = openai.AsyncOpenAI(base_url=base_url, api_key=api_key, timeout=_TIMEOUT_SECONDS)

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
            try:
                response = await self._client.chat.completions.create(
                    model=params.model,
                    max_tokens=params.max_tokens,
                    temperature=params.temperature,
                    messages=messages,
                    tools=tools,
                    tool_choice={"type": "function", "function": {"name": _TOOL_NAME}},
                )
            except openai.InternalServerError as exc:
                # The provider itself failed constructing the tool call server-side
                # (confirmed cause with Ollama + gpt-oss: max_tokens cut the answer
                # off mid-JSON-string) -- never reaches the tool-call/JSON checks
                # below, so it needs its own retry path, nudging toward brevity
                # since a too-long answer is what triggers this.
                last_error = LLMStructuredCallError(
                    f"{call_type}: provider returned {exc.status_code} while generating a response: {exc}"
                )
                messages.append(
                    {
                        "role": "user",
                        "content": "That failed -- keep your answer much shorter this time, then call the tool.",
                    }
                )
                continue
            except openai.APIConnectionError as exc:
                # Covers openai.APITimeoutError too (a subclass of this).
                # Never reached the model at all (or never got a response back in
                # time) -- nothing to add to `messages`, just retry the exact same
                # request. Confirmed in practice: a large local model's cold first
                # inference call while Ollama loads it into memory can be far
                # slower than a warm one, easily a one-off cost rather than a real
                # problem with the request itself.
                last_error = LLMStructuredCallError(f"{call_type}: {type(exc).__name__}: {exc}")
                continue

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
