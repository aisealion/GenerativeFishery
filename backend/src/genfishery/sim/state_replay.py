"""Reconstructs `FisheryState` by replaying a fishery's own event log --
the design `sim/state.py`'s own module docstring always intended ("not
itself persisted -- the event log is; this is reconstructible from replaying
events"), now load-bearing: a process restart (see `sim/engine.py`'s module
docstring on why restarts happen) has to pick the simulation back up exactly
where it left off, not from round 0.

**Whenever a new mechanic changes `FisheryState` in a way this file doesn't
already know how to replay, update this file too** -- state persistence only
understands event types it's been taught to handle; an event type this
function doesn't recognize is silently ignored, which would quietly corrupt
reconstructed state (not crash) if some new mechanic's effect goes
unreplayed.

Known limitation: a migrated-in agent's carried-over personal norm/persona
(set on arrival from the *source* fishery's own state, per
`sim.migration.migrate_random_survivor`) isn't recoverable from this
fishery's own event log alone -- the arrival `MIGRATION` event only records
who arrived and which fishery they came from, not their personal_norm/
persona at the time. A migrated-in agent's norm/persona come back blank
after a restart rather than carried over a second time. Roster presence
(alive, migration_arrival_round) is unaffected -- only that agent's
persona/personal-norm text.
"""

from genfishery.config.fishery_config import FisheryConfig
from genfishery.models.events import Event, EventType
from genfishery.sim.events_sink import EventReader
from genfishery.sim.state import NO_NORM_YET, AgentState, FisheryState


def _apply_event(state: FisheryState, event: Event) -> None:
    payload = event.payload

    if event.type == EventType.FISHERY_INITIALIZED:
        state.persona_descriptions.update(payload.get("persona_descriptions", {}))
        state.persona_types.update(payload.get("persona_types", {}))

    elif event.type == EventType.STRATEGY_DECLARED:
        agent = state.agents.get(event.actor_id)
        if agent is not None:
            agent.last_effort = payload["effort"]

    elif event.type == EventType.HARVEST_RESOLVED:
        consumption = state.config.consumption
        for agent_id, harvest in payload["harvests"].items():
            agent = state.agents.get(agent_id)
            if agent is None:
                continue
            agent.last_harvest = harvest
            agent.payoff += harvest - consumption
            agent.payoff_after_harvest = agent.payoff
        state.stock = payload["regrown_stock"]

    elif event.type == EventType.PENALTY_APPLIED:
        agent = state.agents.get(event.target_id)
        if agent is not None:
            agent.payoff -= payload["amount"]

    elif event.type == EventType.AGENT_STARVED:
        agent = state.agents.get(event.target_id)
        if agent is not None:
            agent.alive = False
            agent.payoff = payload["payoff"]  # authoritative -- overrides accumulation above
        state.starvation_reasons[event.target_id] = payload.get("reason", "starved")

    elif event.type == EventType.PERSONAL_NORM_UPDATED:
        state.agent_norms[event.actor_id] = payload["personal_norm"]

    elif event.type == EventType.NORM_ADOPTED:
        state.group_norm_text = payload["community_proposal"]

    elif event.type == EventType.ROLE_ELECTED:
        state.roles[payload["role_name"]] = payload["agent_id"]

    elif event.type == EventType.FISHERY_COLLAPSED:
        state.collapsed = True
        state.collapse_reasons = list(payload.get("reasons", []))

    elif event.type == EventType.MIGRATION:
        direction = payload.get("direction")
        if direction == "arrival":
            state.agents[event.actor_id] = AgentState(agent_id=event.actor_id)
            state.agent_norms.setdefault(event.actor_id, NO_NORM_YET)
            state.migration_arrival_round[event.actor_id] = event.round
        elif direction == "departure":
            state.agents.pop(event.actor_id, None)
            state.agent_norms.pop(event.actor_id, None)
            state.persona_types.pop(event.actor_id, None)
            state.persona_descriptions.pop(event.actor_id, None)
            for role_name, holder in list(state.roles.items()):
                if holder == event.actor_id:
                    del state.roles[role_name]

    # Every other event type (proposal/vote/councillor-discussion transcript,
    # disclosures, monitor reviews, role-election-called, norm_implemented/
    # norm_implementation_failed -- the SE agent's own code-change outcome,
    # not fishery state) is round-scratch -- informative for memory/UI, but
    # not part of FisheryState itself.


async def reconstruct_state(
    fishery_id: str, events: EventReader, config: FisheryConfig
) -> FisheryState | None:
    """Replays every persisted event for `fishery_id` to rebuild its
    `FisheryState`. Returns None if there's no history at all -- the caller
    should fall back to `FisheryState.initial(config)`, exactly as if this
    fishery had never run before.
    """
    history = await events.list_events(fishery_id)
    if not history:
        return None

    state = FisheryState(
        config=config,
        stock=config.initial_stock,
        agents={aid: AgentState(agent_id=aid) for aid in config.initial_agent_ids},
    )
    for event in history:
        _apply_event(state, event)
    state.round = max(event.round for event in history)
    return state
