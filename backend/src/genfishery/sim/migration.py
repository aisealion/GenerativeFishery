"""Migration on fishery collapse (build spec §7).

Memory is portable because `MemoryBankRegistry` is shared across both
fisheries (keyed by agent_id, owned by neither) -- migrating an agent means
re-keying which fishery's roster they're in, never copying or resetting their
`AgentMemoryBank`. Only fishery-scoped state (payoff, last effort/harvest,
roles) resets; personal belief (`agent_norms`) and memory both carry over,
since both represent who the agent *is*, not fishery-specific bookkeeping.

The migration events themselves are also written into memory (best-effort,
same as `run_memory_write_phase`) -- both sides' `source_llm`/`target_llm`
are optional so tests/callers that don't care about memory can omit them.
"""

import logging
import random

from genfishery.llm.client import LLMClient, LLMStructuredCallError
from genfishery.memory.registry import MemoryBankRegistry
from genfishery.memory.writer import write_observation
from genfishery.models.events import Event, EventType
from genfishery.sim.events_sink import EventSink
from genfishery.sim.observations import render_event_as_observation
from genfishery.sim.state import NO_NORM_YET, AgentState, FisheryState

logger = logging.getLogger(__name__)


async def _write_migration_memory(
    llm: LLMClient,
    memory_registry: MemoryBankRegistry,
    event: Event,
    viewer_id: str,
    round_number: int,
) -> None:
    content = render_event_as_observation(event, viewer_id)
    if content is None:
        return
    bank = memory_registry.get_or_create(viewer_id)
    try:
        await write_observation(llm, bank, content, round=round_number)
    except LLMStructuredCallError:
        logger.warning(
            "migration memory write failed for agent %s (event type %s) -- skipping",
            viewer_id,
            event.type,
        )


async def migrate_random_survivor(
    source: FisheryState,
    target: FisheryState,
    memory_registry: MemoryBankRegistry,
    source_events: EventSink,
    target_events: EventSink,
    *,
    source_llm: LLMClient | None = None,
    target_llm: LLMClient | None = None,
    rng: random.Random | None = None,
) -> str | None:
    """Picks one uniformly random *surviving* agent from `source` (starved
    agents are already removed elsewhere and never reach here) and moves them
    into `target`'s roster immediately -- not at a round boundary. Returns the
    migrated agent_id, or None if `source` has no survivors.
    """
    rng = rng or random.Random()
    survivors = source.alive_agents
    if not survivors:
        return None

    chosen = rng.choice(survivors)
    agent_id = chosen.agent_id
    personal_norm = source.agent_norms.get(agent_id, NO_NORM_YET)
    persona_type = source.persona_types.get(agent_id)
    persona_description = source.persona_descriptions.get(agent_id)

    # Remove from source's roster immediately -- fishery A no longer has any
    # record (live or historical) of them; they're not dead, just relocated.
    del source.agents[agent_id]
    source.agent_norms.pop(agent_id, None)
    source.persona_types.pop(agent_id, None)
    source.persona_descriptions.pop(agent_id, None)
    for role_name, holder in list(source.roles.items()):
        if holder == agent_id:
            del source.roles[role_name]

    # Fresh fishery-scoped state in target: payoff/effort/harvest/roles are
    # this fishery's own bookkeeping, not portable. Personal belief, persona
    # (both represent who the agent *is*, not fishery-specific bookkeeping),
    # and memory (via the shared registry, keyed by agent_id) all carry over
    # intact.
    target.agents[agent_id] = AgentState(agent_id=agent_id)
    target.agent_norms[agent_id] = personal_norm
    if persona_type is not None:
        target.persona_types[agent_id] = persona_type
    if persona_description is not None:
        target.persona_descriptions[agent_id] = persona_description
    target.migration_arrival_round[agent_id] = target.round
    memory_registry.get_or_create(agent_id)  # ensures the bank exists; no-op if already present

    departure_event = Event.create(
        fishery_id=source.config.fishery_id,
        round=source.round,
        phase="migration",
        type=EventType.MIGRATION,
        actor_id=agent_id,
        payload={"direction": "departure", "to_fishery": target.config.fishery_id},
    )
    await source_events.record(departure_event)

    arrival_event = Event.create(
        fishery_id=target.config.fishery_id,
        round=target.round,
        phase="migration",
        type=EventType.MIGRATION,
        actor_id=agent_id,
        payload={"direction": "arrival", "from_fishery": source.config.fishery_id},
    )
    await target_events.record(arrival_event)

    if source_llm is not None:
        # The migrant themself plus everyone left behind in source -- MIGRATION
        # is PUBLIC, so every remaining villager learns "{actor} migrated"
        # (see `render_event_as_observation`), same as any other public event.
        for viewer_id in (agent_id, *(a.agent_id for a in source.alive_agents)):
            await _write_migration_memory(source_llm, memory_registry, departure_event, viewer_id, source.round)

    if target_llm is not None:
        # `target.alive_agents` already includes the migrant (added above).
        for agent in target.alive_agents:
            await _write_migration_memory(target_llm, memory_registry, arrival_event, agent.agent_id, target.round)

    return agent_id


async def handle_collapse_and_migrate(
    source: FisheryState,
    target: FisheryState,
    memory_registry: MemoryBankRegistry,
    source_events: EventSink,
    target_events: EventSink,
    *,
    source_llm: LLMClient | None = None,
    target_llm: LLMClient | None = None,
    rng: random.Random | None = None,
) -> str | None:
    """Call once after a round in which `source.collapsed` became True.

    Collapse is a migration *trigger*, not necessarily a terminal state
    (build spec §7 point 6) -- for resource_depletion/population_loss causes:
    after the one random survivor leaves, if `source` still has agents, it
    un-pauses and keeps running. An underharvest_death collapse is different
    by project decision: it's terminal -- exactly one migrant leaves, then
    `source` stays paused for good, regardless of who's left. No-ops (returns
    None) if `source` wasn't actually collapsed.
    """
    if not source.collapsed:
        return None

    migrated = await migrate_random_survivor(
        source,
        target,
        memory_registry,
        source_events,
        target_events,
        source_llm=source_llm,
        target_llm=target_llm,
        rng=rng,
    )
    if "underharvest_death" not in source.collapse_reasons and source.alive_agents:
        source.collapsed = False
    return migrated
