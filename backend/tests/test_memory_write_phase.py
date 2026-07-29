from genfishery.config.fishery_config import FisheryConfig
from genfishery.config.model_config import LLMCallType
from genfishery.llm.fake_client import FakeLLMClient
from genfishery.memory.importance import ImportanceRating
from genfishery.memory.registry import MemoryBankRegistry
from genfishery.models.events import Event, EventType
from genfishery.sim.engine import run_memory_write_phase
from genfishery.sim.state import FisheryState
from tests.conftest import fake_embedder


def make_config(**overrides) -> FisheryConfig:
    defaults = dict(
        fishery_id="test",
        alpha=0.1,
        r=0.5,
        k=100.0,
        initial_stock=100.0,
        consumption=1.0,
        initial_agent_ids=["a1", "a2"],
        r_min=0.0,
        n_min=1,
        max_rounds=None,
    )
    defaults.update(overrides)
    return FisheryConfig(**defaults)


async def test_memory_write_phase_only_writes_visible_events():
    state = FisheryState.initial(make_config())
    round_events = [
        Event.create(
            fishery_id="test", round=1, phase="strategy", type=EventType.STRATEGY_DECLARED,
            actor_id="a1", payload={"effort": 0.3},
        ),
        Event.create(
            fishery_id="test", round=1, phase="harvest", type=EventType.HARVEST_RESOLVED,
            payload={"harvests": {"a1": 3.0, "a2": 1.0}, "regrown_stock": 90.0},
        ),
    ]
    llm = FakeLLMClient({LLMCallType.IMPORTANCE_RATING: ImportanceRating(score=2.0)})
    registry = MemoryBankRegistry(fake_embedder)

    await run_memory_write_phase(state, llm, registry, round_events)

    a1_memories = [r.content for r in registry.get_or_create("a1").retrieve_recent(10)]
    a2_memories = [r.content for r in registry.get_or_create("a2").retrieve_recent(10)]

    # a1: sees their own private strategy declaration AND the public harvest event.
    assert any("effort 0.30" in m for m in a1_memories)
    assert any("caught 3.00" in m for m in a1_memories)
    # a2: does NOT see a1's actor-only strategy declaration, but does see the public harvest.
    assert not any("effort 0.30" in m for m in a2_memories)
    assert any("caught 1.00" in m for m in a2_memories)


async def test_memory_write_phase_skips_unrenderable_events():
    state = FisheryState.initial(make_config())
    round_events = [
        # A HARVEST_RESOLVED event that doesn't mention either viewer renders
        # to None for both (see `sim.observations.render_event_as_observation`)
        # -- a public event that's nonetheless unrenderable for these viewers.
        Event.create(
            fishery_id="test", round=1, phase="harvest", type=EventType.HARVEST_RESOLVED,
            payload={"harvests": {"a3": 2.0}, "regrown_stock": 95.0},
        ),
        Event.create(
            fishery_id="test", round=1, phase="harvest", type=EventType.HARVEST_RESOLVED,
            payload={"harvests": {"a1": 3.0, "a2": 1.0}, "regrown_stock": 90.0},
        ),
    ]
    llm = FakeLLMClient({LLMCallType.IMPORTANCE_RATING: ImportanceRating(score=1.0)})
    registry = MemoryBankRegistry(fake_embedder)

    await run_memory_write_phase(state, llm, registry, round_events)

    # The unrenderable event contributes nothing; the harvest event still
    # writes exactly one memory each, not zero and not duplicated.
    assert len(registry.get_or_create("a1").retrieve_recent(10)) == 1
    assert len(registry.get_or_create("a2").retrieve_recent(10)) == 1


async def test_memory_write_phase_only_writes_for_alive_agents():
    state = FisheryState.initial(make_config())
    state.agents["a2"].alive = False
    round_events = [
        Event.create(
            fishery_id="test", round=1, phase="harvest", type=EventType.HARVEST_RESOLVED,
            payload={"harvests": {"a1": 3.0, "a2": 1.0}, "regrown_stock": 90.0},
        ),
    ]
    llm = FakeLLMClient({LLMCallType.IMPORTANCE_RATING: ImportanceRating(score=1.0)})
    registry = MemoryBankRegistry(fake_embedder)

    await run_memory_write_phase(state, llm, registry, round_events)

    assert len(registry.get_or_create("a1").retrieve_recent(10)) == 1
    assert "a2" not in registry
