"""Where phase resolution functions append events (build spec §8).

`InMemoryEventSink` is for tests and the (non-persisted) validation sweeps.
`PostgresEventSink` is the real sink -- it inserts into the `events`
hypertable, which fires the `events_notify` trigger (LISTEN/NOTIFY) in the
same transaction as the insert.
"""

from typing import Protocol

from sqlalchemy import insert
from sqlalchemy.ext.asyncio import async_sessionmaker

from genfishery.db.tables import events_table
from genfishery.models.events import Event


class EventSink(Protocol):
    async def record(self, event: Event) -> Event: ...


class InMemoryEventSink:
    def __init__(self) -> None:
        self.events: list[Event] = []

    async def record(self, event: Event) -> Event:
        self.events.append(event)
        return event


class RoundEventCollector:
    """Wraps any EventSink and additionally buffers a copy of everything
    recorded through it, so the round engine can read back "what happened
    this round" (e.g. to write agent memories) without every phase function
    needing to know about that concern -- they just call `.record()` as
    always. Recreated fresh each `run_round` call; nothing to reset.
    """

    def __init__(self, inner: EventSink) -> None:
        self._inner = inner
        self.this_round: list[Event] = []

    async def record(self, event: Event) -> Event:
        recorded = await self._inner.record(event)
        self.this_round.append(recorded)
        return recorded


class PostgresEventSink:
    def __init__(self, sessionmaker: async_sessionmaker) -> None:
        self._sessionmaker = sessionmaker

    async def record(self, event: Event) -> Event:
        # `sessionmaker.begin()` commits on clean exit and rolls back on any
        # exception (including cancellation) automatically -- important here
        # since this runs inside a background asyncio task that gets
        # cancelled on shutdown; a manual execute()+commit() left a
        # dangling "idle in transaction" connection in the pool if cancelled
        # between the two calls, which then poisoned every later use of that
        # pool (see the shared-engine fix in `api/app.py`).
        async with self._sessionmaker.begin() as session:
            result = await session.execute(
                insert(events_table)
                .values(
                    fishery_id=event.fishery_id,
                    round=event.round,
                    phase=event.phase,
                    type=event.type.value,
                    actor_id=event.actor_id,
                    target_id=event.target_id,
                    visibility=event.visibility.value,
                    payload=event.payload,
                )
                .returning(events_table.c.id, events_table.c.created_at)
            )
            row = result.one()
        return event.model_copy(update={"id": row.id, "created_at": row.created_at})
