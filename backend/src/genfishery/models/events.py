"""Append-only event log schema (build spec §8).

Every phase resolution function's only side effect on shared state is to append
rows to this log (plus mutate in-memory `fishery_state`). Prompt assembly and the
frontend both read exclusively from here -- never from an agent's internal
memory objects directly. Each event type carries a *fixed* visibility rule; it is
not a free choice made at construction time.
"""

from collections.abc import Sequence
from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, model_validator


class Visibility(StrEnum):
    PUBLIC = "public"
    ACTOR_ONLY = "actor_only"
    TARGET_ONLY = "target_only"
    ACTOR_AND_TARGET = "actor_and_target"


class EventType(StrEnum):
    STRATEGY_DECLARED = "strategy_declared"
    DISCLOSURE_MADE = "disclosure_made"
    HARVEST_RESOLVED = "harvest_resolved"
    CAP_EXCEEDED = "cap_exceeded"
    PENALTY_APPLIED = "penalty_applied"
    QUOTA_ADJUSTED = "quota_adjusted"
    REDISTRIBUTION_APPLIED = "redistribution_applied"
    AGENT_STARVED = "agent_starved"
    PERSONAL_NORM_UPDATED = "personal_norm_updated"
    PROPOSAL_MADE = "proposal_made"
    VOTE_CAST = "vote_cast"
    VOTE_RESULT = "vote_result"
    OPERATIONALIZATION_PROPOSED = "operationalization_proposed"
    OPERATIONALIZATION_VOTE_CAST = "operationalization_vote_cast"
    OPERATIONALIZATION_RESULT = "operationalization_result"
    NORM_ADOPTED = "norm_adopted"
    NORM_COULD_NOT_COMPILE = "could_not_compile"
    ROLE_ELECTION_CALLED = "role_election_called"
    ROLE_ELECTED = "role_elected"
    MONITOR_REVIEW = "monitor_review"
    FISHERY_COLLAPSED = "fishery_collapsed"
    MIGRATION = "migration"


# The fixed visibility rule per event type. `vote_cast` is deliberately
# actor-only (a ballot is private); `vote_result`, `norm_adopted`, etc. are the
# public facts that follow from it. Types not listed here have no fixed rule
# and must specify `visibility` explicitly at construction time.
VISIBILITY_BY_EVENT_TYPE: dict[EventType, Visibility] = {
    EventType.STRATEGY_DECLARED: Visibility.ACTOR_ONLY,
    EventType.DISCLOSURE_MADE: Visibility.PUBLIC,
    EventType.HARVEST_RESOLVED: Visibility.PUBLIC,
    EventType.CAP_EXCEEDED: Visibility.PUBLIC,
    EventType.PENALTY_APPLIED: Visibility.PUBLIC,
    EventType.QUOTA_ADJUSTED: Visibility.PUBLIC,
    EventType.REDISTRIBUTION_APPLIED: Visibility.PUBLIC,
    EventType.AGENT_STARVED: Visibility.PUBLIC,
    EventType.PERSONAL_NORM_UPDATED: Visibility.ACTOR_ONLY,
    EventType.PROPOSAL_MADE: Visibility.PUBLIC,
    EventType.VOTE_CAST: Visibility.ACTOR_ONLY,
    EventType.VOTE_RESULT: Visibility.PUBLIC,
    EventType.OPERATIONALIZATION_PROPOSED: Visibility.PUBLIC,
    EventType.OPERATIONALIZATION_VOTE_CAST: Visibility.ACTOR_ONLY,
    EventType.OPERATIONALIZATION_RESULT: Visibility.PUBLIC,
    EventType.NORM_ADOPTED: Visibility.PUBLIC,
    EventType.NORM_COULD_NOT_COMPILE: Visibility.PUBLIC,
    EventType.ROLE_ELECTION_CALLED: Visibility.PUBLIC,
    EventType.ROLE_ELECTED: Visibility.PUBLIC,
    EventType.MONITOR_REVIEW: Visibility.PUBLIC,
    EventType.FISHERY_COLLAPSED: Visibility.PUBLIC,
    EventType.MIGRATION: Visibility.PUBLIC,
}


class Event(BaseModel):
    id: int | None = None  # assigned by Postgres (BIGSERIAL) on insert
    fishery_id: str
    round: int
    phase: str
    type: EventType
    actor_id: str | None = None
    target_id: str | None = None
    visibility: Visibility
    payload: dict[str, Any]
    created_at: datetime | None = None  # assigned by Postgres default now()

    @model_validator(mode="after")
    def _enforce_fixed_visibility(self) -> "Event":
        fixed = VISIBILITY_BY_EVENT_TYPE.get(self.type)
        if fixed is not None and self.visibility != fixed:
            raise ValueError(
                f"event type {self.type!r} has a fixed visibility of {fixed!r}, "
                f"got {self.visibility!r}"
            )
        return self

    def is_visible_to(self, viewer_id: str) -> bool:
        """The one place visibility is actually interpreted (build spec §0:
        prompt assembly for any agent is a deterministic function of their own
        memory bank plus whatever public/targeted events they're entitled to
        see -- never an LLM's judgment call).
        """
        if self.visibility == Visibility.PUBLIC:
            return True
        if self.visibility == Visibility.ACTOR_ONLY:
            return self.actor_id == viewer_id
        if self.visibility == Visibility.TARGET_ONLY:
            return self.target_id == viewer_id
        if self.visibility == Visibility.ACTOR_AND_TARGET:
            return viewer_id in (self.actor_id, self.target_id)
        raise ValueError(f"unknown visibility: {self.visibility!r}")

    @classmethod
    def create(
        cls,
        *,
        fishery_id: str,
        round: int,
        phase: str,
        type: EventType,
        payload: dict[str, Any],
        actor_id: str | None = None,
        target_id: str | None = None,
    ) -> "Event":
        """Construct an Event, deriving `visibility` from its fixed per-type rule."""
        visibility = VISIBILITY_BY_EVENT_TYPE.get(type)
        if visibility is None:
            raise ValueError(
                f"event type {type!r} has no fixed visibility rule; "
                "construct it directly with an explicit visibility"
            )
        return cls(
            fishery_id=fishery_id,
            round=round,
            phase=phase,
            type=type,
            actor_id=actor_id,
            target_id=target_id,
            visibility=visibility,
            payload=payload,
        )


def visible_events_for(viewer_id: str, events: Sequence[Event]) -> list[Event]:
    """Filters a sequence of events (e.g. one round's worth) down to those
    `viewer_id` is entitled to see, in the same order they were recorded.
    """
    return [event for event in events if event.is_visible_to(viewer_id)]
