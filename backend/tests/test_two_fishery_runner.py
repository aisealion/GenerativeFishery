"""Tests for step 9: two FisheryRunners running concurrently, migrating
survivors between each other on collapse (build spec §1, §7).
"""

import asyncio

from genfishery.api.runner import FisheryRunner, link_migration
from genfishery.config.fishery_config import FisheryConfig
from genfishery.config.model_config import LLMCallType, ModelConfig
from genfishery.llm.fake_client import FakeLLMClient
from genfishery.memory.importance import ImportanceRating
from genfishery.memory.registry import MemoryBankRegistry
from genfishery.sim.decisions import EffortDecision, ProposalDecision
from genfishery.sim.events_sink import InMemoryEventSink
from genfishery.sim.norm_compiler import NormCompilerOutput
from tests.conftest import fake_embedder


def config_a() -> FisheryConfig:
    # r_min is deliberately far above any stock level this fishery can sustain
    # (K=10, effort fixed at 0.5) -- collapses essentially every round,
    # draining agents into fishery B one at a time.
    return FisheryConfig(
        fishery_id="fishery_a",
        alpha=0.5,
        r=0.5,
        k=10.0,
        initial_stock=5.0,
        consumption=0.0,
        initial_agent_ids=["a1", "a2", "a3"],
        r_min=4.0,
        n_min=1,
        max_rounds=None,
    )


def config_b() -> FisheryConfig:
    return FisheryConfig(
        fishery_id="fishery_b",
        alpha=0.05,
        r=0.5,
        k=200.0,
        initial_stock=200.0,
        consumption=0.0,
        initial_agent_ids=["b1", "b2"],
        r_min=0.0,
        n_min=1,
        max_rounds=None,
    )


def make_llm() -> FakeLLMClient:
    return FakeLLMClient(
        {
            LLMCallType.EFFORT_DECISION: EffortDecision(effort=0.5),
            LLMCallType.IMPORTANCE_RATING: ImportanceRating(score=1.0),
            LLMCallType.PROPOSAL: ProposalDecision(
                personal_norm="ok", community_proposal="Carry on.", operationalization="No change needed."
            ),
            LLMCallType.VOTE: lambda response_model, system, prompt: response_model(chosen_id="1"),
            LLMCallType.NORM_COMPILER: NormCompilerOutput(primitives=[]),
        }
    )


async def _run_until_settled(*runners: FisheryRunner, timeout: float = 5.0) -> None:
    for runner in runners:
        runner.start()
    await asyncio.sleep(timeout)
    for runner in runners:
        await runner.stop()


async def test_migration_drains_fishery_a_into_fishery_b_and_carries_memory():
    llm = make_llm()
    registry = MemoryBankRegistry(fake_embedder)
    events_a = InMemoryEventSink()
    events_b = InMemoryEventSink()
    model_config = ModelConfig.default()

    runner_a = FisheryRunner(config_a(), llm, model_config, events_a, registry, round_interval_seconds=0.0)
    runner_b = FisheryRunner(config_b(), llm, model_config, events_b, registry, round_interval_seconds=0.0)
    link_migration(runner_a, runner_b)

    await _run_until_settled(runner_a, runner_b)

    # Every original fishery_a agent should have ended up in fishery_b.
    assert set(runner_a.state.agents) == set()
    assert {"a1", "a2", "a3"} <= set(runner_b.state.agents)
    # fishery_b's own original agents are still there too.
    assert {"b1", "b2"} <= set(runner_b.state.agents)

    migration_events_a = [e for e in events_a.events if e.type.value == "migration"]
    migration_events_b = [e for e in events_b.events if e.type.value == "migration"]
    assert len(migration_events_a) == 3
    assert len(migration_events_b) == 3
    assert {e.payload["direction"] for e in migration_events_a} == {"departure"}
    assert {e.payload["direction"] for e in migration_events_b} == {"arrival"}

    # Memory carried over intact -- each migrated agent has at least one
    # memory from their time in fishery_a (the shared registry never resets
    # or copies it).
    for agent_id in ("a1", "a2", "a3"):
        assert len(registry.get_or_create(agent_id).retrieve_recent(1000)) > 0


async def test_fishery_b_never_collapses_and_keeps_its_own_clock():
    llm = make_llm()
    registry = MemoryBankRegistry(fake_embedder)
    events_a = InMemoryEventSink()
    events_b = InMemoryEventSink()
    model_config = ModelConfig.default()

    runner_a = FisheryRunner(config_a(), llm, model_config, events_a, registry, round_interval_seconds=0.0)
    runner_b = FisheryRunner(config_b(), llm, model_config, events_b, registry, round_interval_seconds=0.0)
    link_migration(runner_a, runner_b)

    await _run_until_settled(runner_a, runner_b)

    assert runner_b.state.collapsed is False
    # fishery_a ran dry and its task should have stopped on its own (empty
    # roster -> stays paused, per build spec §7 point 6) well before fishery_b,
    # which kept advancing on its own independent clock throughout.
    assert runner_a.state.collapsed is True
    assert runner_b.state.round > 0


async def test_log_dir_writes_one_file_per_fishery(tmp_path):
    llm = make_llm()
    registry = MemoryBankRegistry(fake_embedder)
    events_a = InMemoryEventSink()
    events_b = InMemoryEventSink()
    model_config = ModelConfig.default()

    runner_a = FisheryRunner(
        config_a(), llm, model_config, events_a, registry, round_interval_seconds=0.0, log_dir=tmp_path
    )
    runner_b = FisheryRunner(
        config_b(), llm, model_config, events_b, registry, round_interval_seconds=0.0, log_dir=tmp_path
    )
    link_migration(runner_a, runner_b)

    await _run_until_settled(runner_a, runner_b, timeout=1.0)

    log_a = (tmp_path / "fishery_a.log").read_text()
    log_b = (tmp_path / "fishery_b.log").read_text()
    assert "fishery=fishery_a" in log_a
    assert "fishery=fishery_b" not in log_a
    assert "fishery=fishery_b" in log_b
    assert "fishery=fishery_a" not in log_b
    # Every call type these fisheries actually make should show up somewhere
    # (punishment_decision doesn't -- no DecentralizedPunishment is active in
    # either test config, punishment being opt-in now).
    for call_type in ("effort_decision", "importance_rating"):
        assert call_type in log_a
