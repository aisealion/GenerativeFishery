"""Per-fishery prompt/response logging.

Wraps any `LLMClient` and appends a human-readable record of every
`structured_call` -- system+prompt, and either the parsed response or the
error -- to a single log file, in the exact order calls happen for that
fishery. This covers every call type automatically (effort/punishment/
propose/vote/election/dispute/norm-compiler/importance-rating/reflection)
since they all go through the same `structured_call` entry point -- nothing
else needs to change to get full coverage.

This is a lightweight stand-in for real tracing (build-order step 10:
Langfuse/LangSmith); it doesn't replace that, just gives immediate
plain-text visibility now.
"""

import json
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel

from genfishery.config.model_config import LLMCallType
from genfishery.llm.client import LLMClient

T = TypeVar("T", bound=BaseModel)

_SEPARATOR = "=" * 80


class LoggingLLMClient:
    def __init__(
        self,
        inner: LLMClient,
        log_path: Path,
        *,
        fishery_id: str,
        get_round: Callable[[], int],
    ) -> None:
        self._inner = inner
        self._log_path = log_path
        self._fishery_id = fishery_id
        self._get_round = get_round
        self._log_path.parent.mkdir(parents=True, exist_ok=True)

    async def structured_call(
        self,
        *,
        call_type: LLMCallType,
        system: str,
        prompt: str,
        response_model: type[T],
    ) -> T:
        timestamp = datetime.now(UTC).isoformat()
        header = (
            f"{_SEPARATOR}\n"
            f"[{timestamp}] fishery={self._fishery_id} round={self._get_round()} "
            f"call_type={call_type.value}\n"
            f"--- SYSTEM ---\n{system}\n"
            f"--- PROMPT ---\n{prompt}\n"
        )
        try:
            result = await self._inner.structured_call(
                call_type=call_type, system=system, prompt=prompt, response_model=response_model
            )
        except Exception as exc:
            self._append(f"{header}--- ERROR ---\n{type(exc).__name__}: {exc}\n")
            raise
        response_json = json.dumps(result.model_dump(), default=str)
        self._append(f"{header}--- RESPONSE ---\n{response_json}\n")
        return result

    def _append(self, text: str) -> None:
        with self._log_path.open("a", encoding="utf-8") as f:
            f.write(text)
