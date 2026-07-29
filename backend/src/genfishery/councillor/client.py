"""The fishery councillor: a real `opencode` (opencode.ai) agent, reached over
its HTTP server API (`opencode serve`), that discusses how to operationalize a
just-proposed norm with one fishing agent at a time.

One session per discussion (`start_session` once, then `ask` repeatedly) --
not a fresh `opencode run` subprocess per turn -- because two fisheries can run
concurrently in this codebase (independent asyncio tasks, see
`api/runner.py`'s module docstring) and the CLI's `--continue` only resumes
"the last session" server-wide, which would race across fisheries. An
explicit session id per discussion has no such ambiguity.

Response field names (`TextPart.text`, `{info, parts}`) are taken from
opencode's own SDK type definitions, not exercised against a live server yet
-- `_extract_text`/`_extract_session_id` raise a clear `CouncillorCallError`
with the raw response body on an unexpected shape, so a schema drift surfaces
immediately instead of silently returning empty replies.
"""

from typing import Protocol

import httpx


class CouncillorCallError(RuntimeError):
    """Raised when the councillor server returns an unexpected response shape."""


class CouncillorClient(Protocol):
    async def start_session(self, title: str) -> str:
        """Creates a new discussion session and returns its id."""
        ...

    async def ask(self, session_id: str, message: str) -> str:
        """Sends one message to an existing session and returns the reply text."""
        ...


class HttpCouncillorClient:
    def __init__(
        self,
        base_url: str,
        *,
        agent: str,
        provider_id: str,
        model_id: str,
        timeout: float = 120.0,
    ) -> None:
        self._agent = agent
        self._provider_id = provider_id
        self._model_id = model_id
        self._http = httpx.AsyncClient(base_url=base_url, timeout=timeout)

    async def start_session(self, title: str) -> str:
        response = await self._http.post("/session", json={"title": title})
        response.raise_for_status()
        return _extract_session_id(response.json())

    async def ask(self, session_id: str, message: str) -> str:
        response = await self._http.post(
            f"/session/{session_id}/message",
            json={
                "agent": self._agent,
                "model": {"providerID": self._provider_id, "modelID": self._model_id},
                "parts": [{"type": "text", "text": message}],
            },
        )
        response.raise_for_status()
        return _extract_text(response.json())

    async def aclose(self) -> None:
        await self._http.aclose()


def _extract_session_id(data: dict) -> str:
    session_id = data.get("id")
    if not isinstance(session_id, str):
        raise CouncillorCallError(f"POST /session response had no string 'id' field: {data!r}")
    return session_id


def _extract_text(data: dict) -> str:
    parts = data.get("parts")
    if not isinstance(parts, list):
        raise CouncillorCallError(f"session message response had no 'parts' list: {data!r}")
    text = "".join(part.get("text", "") for part in parts if part.get("type") == "text")
    if not text:
        raise CouncillorCallError(f"session message response had no text part: {data!r}")
    return text
