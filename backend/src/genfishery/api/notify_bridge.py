"""Postgres LISTEN/NOTIFY -> asyncio pub/sub (build spec §1, §9).

A dedicated asyncpg connection LISTENs on the `events` channel (the trigger
from the initial migration fires NOTIFY in the same transaction as each
insert). Each notification is a compact JSON payload (id/fishery_id/round/
phase/type/actor_id/target_id/visibility -- deliberately not `payload`, see
the migration's own comment on the 8000-byte NOTIFY limit); this bridge fans
it out to whichever WebSocket connections are subscribed to that fishery_id.
"""

import asyncio
import json
from collections import defaultdict

import asyncpg


def to_asyncpg_dsn(sqlalchemy_url: str) -> str:
    """asyncpg.connect() wants `postgresql://...`, not SQLAlchemy's
    `postgresql+asyncpg://...`.
    """
    return sqlalchemy_url.replace("postgresql+asyncpg://", "postgresql://", 1)


class NotifyBridge:
    def __init__(self, dsn: str) -> None:
        self._dsn = to_asyncpg_dsn(dsn)
        self._conn: asyncpg.Connection | None = None
        self._subscribers: dict[str, set[asyncio.Queue]] = defaultdict(set)

    async def start(self) -> None:
        self._conn = await asyncpg.connect(self._dsn)
        await self._conn.add_listener("events", self._on_notify)

    async def stop(self) -> None:
        if self._conn is not None:
            await self._conn.remove_listener("events", self._on_notify)
            await self._conn.close()
            self._conn = None

    def _on_notify(self, connection, pid, channel, payload: str) -> None:
        data = json.loads(payload)
        fishery_id = data.get("fishery_id")
        for queue in self._subscribers.get(fishery_id, ()):
            queue.put_nowait(data)

    def subscribe(self, fishery_id: str) -> asyncio.Queue:
        queue: asyncio.Queue = asyncio.Queue()
        self._subscribers[fishery_id].add(queue)
        return queue

    def unsubscribe(self, fishery_id: str, queue: asyncio.Queue) -> None:
        self._subscribers[fishery_id].discard(queue)
