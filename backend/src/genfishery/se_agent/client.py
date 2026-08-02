"""The fishery SE agent: a real `opencode` (opencode.ai) agent, reached over
its HTTP server API (`opencode serve`), with two jobs -- discussing how to
operationalize a just-proposed norm with one fishing agent at a time
(`ask`), and, for whichever proposal wins the community's vote, implementing
it as an actual code change under `backend/`, committed with git
(`implement_norm`).

One session per discussion (`start_session` once, then `ask`/`implement_norm`
repeatedly within it) -- not a fresh `opencode run` subprocess per turn --
because two fisheries can run concurrently in this codebase (independent
asyncio tasks, see `api/runner.py`'s module docstring) and the CLI's
`--continue` only resumes "the last session" server-wide, which would race
across fisheries. An explicit session id per discussion has no such
ambiguity.

Response field names (`TextPart.text`, `{info, parts}`) are taken from
opencode's own SDK type definitions, not exercised against a live server yet
-- `_extract_text`/`_extract_session_id` raise a clear `SEAgentCallError`
with the raw response body on an unexpected shape, so a schema drift surfaces
immediately instead of silently returning empty replies.

`implement_norm`'s success/failure is never taken on the agent's own say-so
-- an LLM's text reply isn't a reliable signal that it actually committed
anything. It's verified by checking whether `git`'s HEAD commit in `repo_dir`
actually changed, the same way this codebase avoids trusting LLM judgment for
any other structural decision.
"""

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import httpx


class SEAgentCallError(RuntimeError):
    """Raised when the SE agent server returns an unexpected response shape."""


@dataclass(frozen=True)
class ImplementResult:
    success: bool
    reply: str
    commit_sha: str | None


class SEAgentClient(Protocol):
    async def start_session(self, title: str) -> str:
        """Creates a new discussion session and returns its id."""
        ...

    async def ask(self, session_id: str, message: str) -> str:
        """Sends one message to an existing session and returns the reply text."""
        ...

    async def implement_norm(self, session_id: str, instructions: str) -> ImplementResult:
        """Tells the agent to implement a norm as a real code change, run the
        test suite, and commit. Success is verified via git, not the reply text.
        """
        ...


def _git_head(repo_dir: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_dir), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    return result.stdout.strip()


class HttpSEAgentClient:
    def __init__(
        self,
        base_url: str,
        *,
        agent: str,
        provider_id: str,
        model_id: str,
        repo_dir: str | Path,
        # A plain chat turn still goes through opencode's own agent loop
        # (system prompt + tool use) against the same shared, sometimes-slow
        # Ollama backend genfishery's own calls hit cold-start/load timeouts
        # on (see openai_compatible_client.py) -- confirmed too tight in
        # practice at 120s, so the default here is far more generous.
        timeout: float = 600.0,
    ) -> None:
        self._agent = agent
        self._provider_id = provider_id
        self._model_id = model_id
        self._repo_dir = Path(repo_dir)
        # Implementing a norm involves real edits + running a test suite,
        # well beyond a normal chat reply -- a much longer timeout than the
        # default is needed so this isn't mistaken for a hung request.
        self._http = httpx.AsyncClient(base_url=base_url, timeout=timeout)
        self._implement_timeout = max(timeout, 1800.0)

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

    async def implement_norm(self, session_id: str, instructions: str) -> ImplementResult:
        before = _git_head(self._repo_dir)
        response = await self._http.post(
            f"/session/{session_id}/message",
            json={
                "agent": self._agent,
                "model": {"providerID": self._provider_id, "modelID": self._model_id},
                "parts": [{"type": "text", "text": instructions}],
            },
            timeout=self._implement_timeout,
        )
        response.raise_for_status()
        reply = _extract_text(response.json())
        after = _git_head(self._repo_dir)
        success = after is not None and after != before
        return ImplementResult(success=success, reply=reply, commit_sha=after if success else None)

    async def aclose(self) -> None:
        await self._http.aclose()


def _extract_session_id(data: dict) -> str:
    session_id = data.get("id")
    if not isinstance(session_id, str):
        raise SEAgentCallError(f"POST /session response had no string 'id' field: {data!r}")
    return session_id


def _extract_text(data: dict) -> str:
    parts = data.get("parts")
    if not isinstance(parts, list):
        raise SEAgentCallError(f"session message response had no 'parts' list: {data!r}")
    text = "".join(part.get("text", "") for part in parts if part.get("type") == "text")
    if not text:
        raise SEAgentCallError(f"session message response had no text part: {data!r}")
    return text
