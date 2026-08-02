"""Per-fishery SE-agent logging, mirroring `llm/logging_client.py`'s exact
append-only pattern -- one plain-text log file per fishery, kept separate
from the LLM call log since this is a different external service (opencode,
not the fishing agents' own LLM provider).
"""

from datetime import UTC, datetime
from pathlib import Path

from genfishery.se_agent.client import ImplementResult, SEAgentClient

_SEPARATOR = "=" * 80


class LoggingSEAgentClient:
    def __init__(self, inner: SEAgentClient, log_path: Path, *, fishery_id: str) -> None:
        self._inner = inner
        self._log_path = log_path
        self._fishery_id = fishery_id
        self._log_path.parent.mkdir(parents=True, exist_ok=True)

    async def start_session(self, title: str) -> str:
        try:
            session_id = await self._inner.start_session(title)
        except Exception as exc:
            self._append(
                f"{_SEPARATOR}\n[{self._timestamp()}] fishery={self._fishery_id} "
                f"start_session title={title!r}\n--- ERROR ---\n{type(exc).__name__}: {exc}\n"
            )
            raise
        self._append(
            f"{_SEPARATOR}\n[{self._timestamp()}] fishery={self._fishery_id} "
            f"start_session title={title!r} -> session_id={session_id}\n"
        )
        return session_id

    async def ask(self, session_id: str, message: str) -> str:
        header = (
            f"{_SEPARATOR}\n[{self._timestamp()}] fishery={self._fishery_id} "
            f"session_id={session_id}\n--- MESSAGE ---\n{message}\n"
        )
        try:
            reply = await self._inner.ask(session_id, message)
        except Exception as exc:
            self._append(f"{header}--- ERROR ---\n{type(exc).__name__}: {exc}\n")
            raise
        self._append(f"{header}--- REPLY ---\n{reply}\n")
        return reply

    async def implement_norm(self, session_id: str, instructions: str) -> ImplementResult:
        header = (
            f"{_SEPARATOR}\n[{self._timestamp()}] fishery={self._fishery_id} "
            f"session_id={session_id}\n--- IMPLEMENT ---\n{instructions}\n"
        )
        try:
            result = await self._inner.implement_norm(session_id, instructions)
        except Exception as exc:
            self._append(f"{header}--- ERROR ---\n{type(exc).__name__}: {exc}\n")
            raise
        self._append(
            f"{header}--- RESULT ---\nsuccess={result.success} commit_sha={result.commit_sha}\n"
            f"--- REPLY ---\n{result.reply}\n"
        )
        return result

    @staticmethod
    def _timestamp() -> str:
        return datetime.now(UTC).isoformat()

    def _append(self, text: str) -> None:
        with self._log_path.open("a", encoding="utf-8") as f:
            f.write(text)
