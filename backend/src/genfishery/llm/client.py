"""LLM client interface (build spec §0, §2).

Every agent decision and the NormCompiler go through `structured_call`: it
always returns a validated instance of a fixed pydantic model, never free text.
There is exactly one implementation surface here -- `AnthropicLLMClient` for
real runs, `FakeLLMClient` for tests -- so application code never talks to the
Anthropic SDK directly and never parses free-form text into behavior.
"""

from typing import Protocol, TypeVar

from pydantic import BaseModel

from genfishery.config.model_config import LLMCallType

T = TypeVar("T", bound=BaseModel)


class LLMStructuredCallError(RuntimeError):
    """Raised when a call doesn't yield a valid, schema-conforming response.

    Callers must not catch this to improvise a substitute answer -- the one
    sanctioned reaction is what NormCompiler does: log the failure as data
    (`could_not_compile`) and treat the input as descriptive-only.
    """


class LLMClient(Protocol):
    async def structured_call(
        self,
        *,
        call_type: LLMCallType,
        system: str,
        prompt: str,
        response_model: type[T],
    ) -> T:
        """Issue one structured-output call and return a validated `response_model`.

        Implementations must never fall back to free-text parsing or improvise
        a response on schema-validation failure -- they raise, and the caller
        decides how to handle it (e.g. NormCompiler logs `could_not_compile`).
        """
        ...
