"""Build-order step 7: manual two-fishery migration demo.

Runs fishery A (deliberately configured to collapse quickly) round by round;
whenever it collapses, `handle_collapse_and_migrate` picks one random survivor
and moves them into fishery B, which keeps running independently. This is the
"manual trigger for testing" version the build order asks for -- there is no
async orchestration of two live, independently-clocked fisheries yet (that's
step 9); here a single script just drives both, checking collapse after each
of fishery A's rounds.

Uses rule-based (non-LLM) agents for effort/punishment (see
scripts/validate_gupta_replication.py for why) and a FakeLLMClient for the
memory-write phase's importance rating, so this runs standalone with no API
key. Real runs would pass an AnthropicLLMClient instead.

Usage: uv run python scripts/manual_migration_demo.py
"""

import asyncio

from genfishery.config.fishery_config import FisheryConfig
from genfishery.config.model_config import LLMCallType
from genfishery.llm.fake_client import FakeLLMClient
from genfishery.memory.embedder import get_embedder
from genfishery.memory.importance import ImportanceRating
from genfishery.memory.registry import MemoryBankRegistry
from genfishery.sim.engine import run_round
from genfishery.sim.events_sink import InMemoryEventSink
from genfishery.sim.migration import handle_collapse_and_migrate
from genfishery.sim.policies import RuleBasedDecisionSource
from genfishery.sim.state import FisheryState


def fishery_a_config() -> FisheryConfig:
    # Small stock + aggressive effort dynamics -> collapses within a handful
    # of rounds, so migration triggers quickly for this demo.
    return FisheryConfig(
        fishery_id="fishery_a",
        alpha=0.3,
        r=0.3,
        k=30.0,
        initial_stock=30.0,
        consumption=1.0,
        initial_agent_ids=["a1", "a2", "a3", "a4", "a5"],
        r_min=5.0,
        n_min=4,  # collapses as soon as a single agent is lost
        max_rounds=30,
    )


def fishery_b_config() -> FisheryConfig:
    return FisheryConfig(
        fishery_id="fishery_b",
        alpha=0.08,
        r=0.5,
        k=100.0,
        initial_stock=100.0,
        consumption=1.0,
        initial_agent_ids=["b1", "b2", "b3"],
        r_min=0.0,
        n_min=1,
        max_rounds=None,
    )


async def main() -> None:
    state_a = FisheryState.initial(fishery_a_config())
    state_b = FisheryState.initial(fishery_b_config())

    decisions_a = RuleBasedDecisionSource(state_a.config.initial_agent_ids, seed=1)
    decisions_b = RuleBasedDecisionSource(state_b.config.initial_agent_ids, seed=2)

    events_a = InMemoryEventSink()
    events_b = InMemoryEventSink()

    memory_llm = FakeLLMClient({LLMCallType.IMPORTANCE_RATING: ImportanceRating(score=3.0)})
    registry = MemoryBankRegistry(get_embedder())

    print("Running fishery A until it collapses (or hits max_rounds)...")
    while not state_a.collapsed and (
        state_a.config.max_rounds is None or state_a.round < state_a.config.max_rounds
    ):
        await run_round(
            state_a, decisions_a, events_a, llm=memory_llm, memory_registry=registry
        )
        print(
            f"  [A] round {state_a.round}: stock={state_a.stock:.1f}, "
            f"alive={[a.agent_id for a in state_a.alive_agents]}"
        )
        if state_a.collapsed:
            print(f"  [A] COLLAPSED at round {state_a.round} (stock={state_a.stock:.1f}).")
            migrated = await handle_collapse_and_migrate(
                state_a, state_b, registry, events_a, events_b
            )
            if migrated is None:
                print("  [A] no survivors to migrate.")
            else:
                bank = registry.get_or_create(migrated)
                print(
                    f"  >>> {migrated} migrated to fishery B, carrying "
                    f"{len(bank.retrieve_recent(1000))} memories."
                )
            print(
                f"  [A] after migration: alive={[a.agent_id for a in state_a.alive_agents]}, "
                f"paused={state_a.collapsed}"
            )

    print("\nRunning 5 more rounds of fishery B to confirm the migrant participates normally...")
    for _ in range(5):
        if not state_b.alive_agents:
            break
        await run_round(state_b, decisions_b, events_b, llm=memory_llm, memory_registry=registry)
        print(
            f"  [B] round {state_b.round}: stock={state_b.stock:.1f}, "
            f"alive={[a.agent_id for a in state_b.alive_agents]}"
        )

    print("\nMigration events logged:")
    for event in events_a.events + events_b.events:
        if event.type.value == "migration":
            print(f"  {event.fishery_id} round {event.round}: {event.payload}")


if __name__ == "__main__":
    asyncio.run(main())
