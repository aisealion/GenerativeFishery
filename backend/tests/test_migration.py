import random

from genfishery.config.fishery_config import FisheryConfig
from genfishery.config.model_config import LLMCallType
from genfishery.llm.fake_client import FakeLLMClient
from genfishery.memory.importance import ImportanceRating
from genfishery.memory.registry import MemoryBankRegistry
from genfishery.models.events import EventType
from genfishery.sim.events_sink import InMemoryEventSink
from genfishery.sim.migration import handle_collapse_and_migrate, migrate_random_survivor
from genfishery.sim.state import FisheryState
from tests.conftest import fake_embedder


def make_llm() -> FakeLLMClient:
    return FakeLLMClient({LLMCallType.IMPORTANCE_RATING: ImportanceRating(score=5.0)})


def make_config(fishery_id: str, **overrides) -> FisheryConfig:
    defaults = dict(
        fishery_id=fishery_id,
        alpha=0.1,
        r=0.5,
        k=100.0,
        initial_stock=100.0,
        consumption=1.0,
        initial_agent_ids=["a1", "a2", "a3"],
        r_min=0.0,
        n_min=1,
        max_rounds=None,
    )
    defaults.update(overrides)
    return FisheryConfig(**defaults)


async def test_migrate_random_survivor_moves_agent_between_rosters():
    source = FisheryState.initial(make_config("fishery_a"))
    target = FisheryState.initial(make_config("fishery_b", initial_agent_ids=["b1", "b2"]))
    registry = MemoryBankRegistry(fake_embedder)
    source_events = InMemoryEventSink()
    target_events = InMemoryEventSink()

    migrated = await migrate_random_survivor(
        source, target, registry, source_events, target_events, rng=random.Random(0)
    )

    assert migrated in {"a1", "a2", "a3"}
    assert migrated not in source.agents
    assert migrated not in source.agent_norms
    assert migrated in target.agents
    assert target.agents[migrated].payoff == 0.0
    assert target.agents[migrated].alive is True


async def test_migrate_random_survivor_carries_personal_norm_and_memory_bank_identity():
    source = FisheryState.initial(make_config("fishery_a"))
    source.agent_norms["a1"] = "Fish moderately to preserve the lake."
    registry = MemoryBankRegistry(fake_embedder)
    bank_before = registry.get_or_create("a1")
    bank_before.add_memory("I remember something important.", importance=8.0, kind="observation", round=1)

    target = FisheryState.initial(make_config("fishery_b", initial_agent_ids=["b1"]))
    source_events = InMemoryEventSink()
    target_events = InMemoryEventSink()

    # Force a1 specifically by using a roster where a1 is the only alive agent.
    source.agents["a2"].alive = False
    source.agents["a3"].alive = False

    migrated = await migrate_random_survivor(source, target, registry, source_events, target_events)
    assert migrated == "a1"
    assert target.agent_norms["a1"] == "Fish moderately to preserve the lake."

    bank_after = registry.get_or_create("a1")
    assert bank_after is bank_before  # same object -- memory was never copied or reset
    assert len(bank_after.retrieve_recent(10)) == 1


async def test_migrate_random_survivor_carries_persona_identity():
    source = FisheryState.initial(make_config("fishery_a"))
    source.persona_types["a1"] = "selfish"
    source.persona_descriptions["a1"] = "Maximize your catch while the fish are abundant"
    source.agents["a2"].alive = False
    source.agents["a3"].alive = False
    registry = MemoryBankRegistry(fake_embedder)
    target = FisheryState.initial(make_config("fishery_b", initial_agent_ids=["b1"]))

    migrated = await migrate_random_survivor(
        source, target, registry, InMemoryEventSink(), InMemoryEventSink()
    )

    assert migrated == "a1"
    assert target.persona_types["a1"] == "selfish"
    assert target.persona_descriptions["a1"] == "Maximize your catch while the fish are abundant"
    assert "a1" not in source.persona_types
    assert "a1" not in source.persona_descriptions


async def test_migrate_random_survivor_stamps_arrival_round_in_target_clock():
    source = FisheryState.initial(make_config("fishery_a", initial_agent_ids=["a1"]))
    target = FisheryState.initial(make_config("fishery_b", initial_agent_ids=["b1"]))
    target.round = 12  # target's own independent clock, unrelated to source's
    registry = MemoryBankRegistry(fake_embedder)

    migrated = await migrate_random_survivor(
        source, target, registry, InMemoryEventSink(), InMemoryEventSink()
    )

    assert target.migration_arrival_round[migrated] == 12


async def test_migrate_random_survivor_logs_migration_events_in_both_fisheries():
    source = FisheryState.initial(make_config("fishery_a", initial_agent_ids=["a1"]))
    target = FisheryState.initial(make_config("fishery_b", initial_agent_ids=["b1"]))
    registry = MemoryBankRegistry(fake_embedder)
    source_events = InMemoryEventSink()
    target_events = InMemoryEventSink()

    await migrate_random_survivor(source, target, registry, source_events, target_events)

    source_event = next(e for e in source_events.events if e.type == EventType.MIGRATION)
    target_event = next(e for e in target_events.events if e.type == EventType.MIGRATION)
    assert source_event.visibility.value == "public"
    assert source_event.payload == {"direction": "departure", "to_fishery": "fishery_b"}
    assert target_event.payload == {"direction": "arrival", "from_fishery": "fishery_a"}
    assert source_event.actor_id == target_event.actor_id == "a1"


async def test_migrate_random_survivor_without_llm_writes_no_memories():
    """Default/no-op behavior when callers don't care about memory (matches
    every pre-existing test above, which omits source_llm/target_llm)."""
    source = FisheryState.initial(make_config("fishery_a", initial_agent_ids=["a1"]))
    target = FisheryState.initial(make_config("fishery_b", initial_agent_ids=["b1"]))
    registry = MemoryBankRegistry(fake_embedder)

    await migrate_random_survivor(source, target, registry, InMemoryEventSink(), InMemoryEventSink())

    assert registry.get_or_create("a1").retrieve_recent(10) == []
    assert registry.get_or_create("b1").retrieve_recent(10) == []


async def test_migrate_random_survivor_writes_departure_and_arrival_memories():
    source = FisheryState.initial(make_config("fishery_a", initial_agent_ids=["a1", "a2"]))
    target = FisheryState.initial(make_config("fishery_b", initial_agent_ids=["b1"]))
    registry = MemoryBankRegistry(fake_embedder)

    migrated = await migrate_random_survivor(
        source,
        target,
        registry,
        InMemoryEventSink(),
        InMemoryEventSink(),
        source_llm=make_llm(),
        target_llm=make_llm(),
        rng=random.Random(0),
    )

    # The migrant remembers both leaving and arriving.
    migrant_memories = [r.content for r in registry.get_or_create(migrated).retrieve_recent(10)]
    assert any("I left for another fishery" in c for c in migrant_memories)
    assert any("I arrived here from another fishery" in c for c in migrant_memories)

    # Villagers left behind in source learn the migrant departed.
    bystander_id = next(a for a in ["a1", "a2"] if a != migrated)
    bystander_memories = [r.content for r in registry.get_or_create(bystander_id).retrieve_recent(10)]
    assert any(f"{migrated} migrated to another fishery" in c for c in bystander_memories)

    # Existing villagers in target are told a new fisherman arrived.
    b1_memories = [r.content for r in registry.get_or_create("b1").retrieve_recent(10)]
    assert any(f"A new fisherman, {migrated}, has arrived" in c for c in b1_memories)


async def test_migrate_random_survivor_returns_none_when_source_has_no_survivors():
    source = FisheryState.initial(make_config("fishery_a", initial_agent_ids=["a1"]))
    source.agents["a1"].alive = False
    target = FisheryState.initial(make_config("fishery_b", initial_agent_ids=["b1"]))
    registry = MemoryBankRegistry(fake_embedder)

    migrated = await migrate_random_survivor(
        source, target, registry, InMemoryEventSink(), InMemoryEventSink()
    )
    assert migrated is None


async def test_handle_collapse_and_migrate_unpauses_source_if_survivors_remain():
    source = FisheryState.initial(make_config("fishery_a", initial_agent_ids=["a1", "a2"]))
    source.collapsed = True
    target = FisheryState.initial(make_config("fishery_b", initial_agent_ids=["b1"]))
    registry = MemoryBankRegistry(fake_embedder)

    migrated = await handle_collapse_and_migrate(
        source, target, registry, InMemoryEventSink(), InMemoryEventSink()
    )

    assert migrated is not None
    assert source.collapsed is False  # one agent remains -- collapse was a trigger, not terminal
    assert len(source.alive_agents) == 1


async def test_handle_collapse_and_migrate_stays_paused_when_source_becomes_empty():
    source = FisheryState.initial(make_config("fishery_a", initial_agent_ids=["a1"]))
    source.collapsed = True
    target = FisheryState.initial(make_config("fishery_b", initial_agent_ids=["b1"]))
    registry = MemoryBankRegistry(fake_embedder)

    migrated = await handle_collapse_and_migrate(
        source, target, registry, InMemoryEventSink(), InMemoryEventSink()
    )

    assert migrated == "a1"
    assert source.alive_agents == []
    assert source.collapsed is True  # empty -- stays paused, no auto-restart


async def test_handle_collapse_and_migrate_stays_paused_for_good_on_underharvest_death():
    """An underharvest_death collapse is terminal by project decision: exactly
    one migrant leaves and the fishery stays paused even though survivors
    remain -- unlike resource_depletion/population_loss, which un-pause."""
    source = FisheryState.initial(make_config("fishery_a", initial_agent_ids=["a1", "a2", "a3"]))
    source.collapsed = True
    source.collapse_reasons = ["underharvest_death"]
    target = FisheryState.initial(make_config("fishery_b", initial_agent_ids=["b1"]))
    registry = MemoryBankRegistry(fake_embedder)

    migrated = await handle_collapse_and_migrate(
        source, target, registry, InMemoryEventSink(), InMemoryEventSink()
    )

    assert migrated is not None
    assert len(source.alive_agents) == 2  # two survivors remain in source...
    assert source.collapsed is True  # ...but it stays paused anyway


async def test_handle_collapse_and_migrate_no_op_if_not_collapsed():
    source = FisheryState.initial(make_config("fishery_a"))
    assert source.collapsed is False
    target = FisheryState.initial(make_config("fishery_b", initial_agent_ids=["b1"]))
    registry = MemoryBankRegistry(fake_embedder)

    migrated = await handle_collapse_and_migrate(
        source, target, registry, InMemoryEventSink(), InMemoryEventSink()
    )
    assert migrated is None
    assert len(source.alive_agents) == 3  # untouched
