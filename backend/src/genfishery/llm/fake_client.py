"""Deterministic fake LLM client for tests (build spec: mockable client answer).

Tests script exact responses per call type instead of hitting the real
Anthropic API, so unit tests and the step-2 paper-validation run stay
reproducible and free. Scripted responses are still validated against the
`response_model` the caller asked for, so a test with a mis-shaped script
fails the same way a real malformed tool_use response would.
"""

from collections import deque
from collections.abc import Callable, Iterable
from typing import TypeVar

from pydantic import BaseModel

from genfishery.config.model_config import LLMCallType
from genfishery.llm.client import LLMStructuredCallError

T = TypeVar("T", bound=BaseModel)

# A single response reused for every call of that type, a sequence consumed
# one-at-a-time (in order) across successive calls, or a responder callable
# invoked with the *actual* `response_model` the caller asked for -- needed
# for call types like VOTE whose response schema is built fresh per call
# (`build_vote_response_model`), so a pre-built instance would never satisfy
# `isinstance(result, response_model)` against that call's own model class.
Script = dict[
    LLMCallType,
    BaseModel | Iterable[BaseModel] | Callable[[type[BaseModel], str, str], BaseModel],
]


class FakeLLMClient:
    def __init__(self, script: Script) -> None:
        self._fixed: dict[LLMCallType, BaseModel] = {}
        self._queues: dict[LLMCallType, deque[BaseModel]] = {}
        self._responders: dict[LLMCallType, Callable[[type[BaseModel], str, str], BaseModel]] = {}
        for call_type, entry in script.items():
            if isinstance(entry, BaseModel):
                self._fixed[call_type] = entry
            elif callable(entry):
                self._responders[call_type] = entry
            else:
                self._queues[call_type] = deque(entry)
        self.calls: list[tuple[LLMCallType, str, str]] = []

    async def structured_call(
        self,
        *,
        call_type: LLMCallType,
        system: str,
        prompt: str,
        response_model: type[T],
    ) -> T:
        self.calls.append((call_type, system, prompt))

        if call_type in self._fixed:
            result = self._fixed[call_type]
        elif call_type in self._responders:
            result = self._responders[call_type](response_model, system, prompt)
        elif call_type in self._queues:
            queue = self._queues[call_type]
            if not queue:
                raise LLMStructuredCallError(
                    f"{call_type}: fake script exhausted (no more scripted responses)"
                )
            result = queue.popleft()
        else:
            raise LLMStructuredCallError(f"{call_type}: no script provided for this call type")

        if not isinstance(result, response_model):
            raise LLMStructuredCallError(
                f"{call_type}: scripted response is {type(result).__name__}, "
                f"caller requested {response_model.__name__}"
            )
        return result
