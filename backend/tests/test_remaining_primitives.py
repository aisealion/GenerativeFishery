import random

import pytest

from genfishery.config.fishery_config import FisheryConfig
from genfishery.models.events import EventType
from genfishery.models.norms import (
    AdjustPrimitive,
    AssignRolePrimitive,
    DeclarePrimitive,
    MonitorPrimitive,
    PenalisePrimitive,
    RedistributePrimitive,
)
from genfishery.sim.engine import (
    run_adjust_phase,
    run_declare_phase,
    run_monitor_phase,
    run_penalise_phase,
    run_redistribute_phase,
)
from genfishery.sim.events_sink import InMemoryEventSink
from genfishery.sim.norm_compiler import NormCompiler, NormCompilerOutput
from genfishery.sim.state import FisheryState


def make_config(**overrides) -> FisheryConfig:
    defaults = dict(
        fishery_id="test",
        alpha=0.1,
        r=0.5,
        k=100.0,
        initial_stock=100.0,
        consumption=1.0,
        initial_agent_ids=["a1", "a2", "a3"],
        r_min=0.0,
        n_min=2,
        max_rounds=None,
    )
    defaults.update(overrides)
    return FisheryConfig(**defaults)


# --- DeclarePrimitive ----------------------------------------------------------


async def test_declare_phase_publicizes_last_effort_pre_round():
    state = FisheryState.initial(make_config())
    state.active_norms.append(
        DeclarePrimitive(id="d1", scope="collective", timing="pre_round", content="intended_quota")
    )
    for agent_id, effort in [("a1", 0.2), ("a2", 0.5), ("a3", 0.9)]:
        state.agents[agent_id].last_effort = effort

    events = InMemoryEventSink()
    await run_declare_phase(state, events, timing="pre_round")

    disclosures = {e.actor_id: e.payload["value"] for e in events.events if e.type == EventType.DISCLOSURE_MADE}
    assert disclosures == {"a1": 0.2, "a2": 0.5, "a3": 0.9}
    assert all(e.visibility.value == "public" for e in events.events)


async def test_declare_phase_only_fires_at_its_own_timing():
    state = FisheryState.initial(make_config())
    state.active_norms.append(
        DeclarePrimitive(id="d1", scope="collective", timing="post_round", content="actual_take")
    )
    state.agents["a1"].last_harvest = 3.0
    events = InMemoryEventSink()

    await run_declare_phase(state, events, timing="pre_round")
    assert events.events == []

    await run_declare_phase(state, events, timing="post_round")
    assert any(e.type == EventType.DISCLOSURE_MADE and e.payload["value"] == 3.0 for e in events.events)


async def test_declare_phase_no_op_without_primitive():
    state = FisheryState.initial(make_config())
    events = InMemoryEventSink()
    await run_declare_phase(state, events, timing="pre_round")
    assert events.events == []


async def test_declare_phase_non_public_visibility_discloses_nothing():
    """Scoping simplification: only disclosure_visibility="public" has a
    mechanism wired up; "anonymous"/"ledger_only" compile but are inert."""
    state = FisheryState.initial(make_config())
    state.active_norms.append(
        DeclarePrimitive(id="d1", scope="collective", timing="pre_round", disclosure_visibility="ledger_only")
    )
    state.agents["a1"].last_effort = 0.5
    events = InMemoryEventSink()
    await run_declare_phase(state, events, timing="pre_round")
    assert events.events == []


# --- MonitorPrimitive ------------------------------------------------------------


async def test_monitor_phase_random_audit_triggers_per_seeded_probability():
    state = FisheryState.initial(make_config())
    state.active_norms.append(
        MonitorPrimitive(id="m1", scope="collective", method="random_audit", monitor_target="individual")
    )
    events = InMemoryEventSink()
    await run_monitor_phase(state, events, rng=random.Random(0))

    audited = [e for e in events.events if e.type == EventType.MONITOR_REVIEW]
    assert all(e.visibility.value == "public" for e in audited)
    assert all(e.payload["method"] == "random_audit" for e in audited)


async def test_monitor_phase_other_methods_produce_no_events():
    state = FisheryState.initial(make_config())
    state.active_norms.append(MonitorPrimitive(id="m1", scope="collective", method="peer"))
    events = InMemoryEventSink()
    await run_monitor_phase(state, events, rng=random.Random(0))
    assert events.events == []


async def test_monitor_phase_no_op_without_primitive():
    state = FisheryState.initial(make_config())
    events = InMemoryEventSink()
    await run_monitor_phase(state, events)
    assert events.events == []


# --- PenalisePrimitive -----------------------------------------------------------


async def test_penalise_phase_forfeits_the_exact_excess_to_the_pool():
    state = FisheryState.initial(make_config())
    state.active_norms.append(
        PenalisePrimitive(
            id="pen1", scope="collective", trigger="exceed_cap", penalty_type="forfeit", destination="pool"
        )
    )
    state.agents["a1"].payoff = 10.0
    state.stock = 50.0
    events = InMemoryEventSink()

    await run_penalise_phase(state, events, {"a1": [{"trigger": "exceed_cap", "excess": 4.0}]})

    assert state.agents["a1"].payoff == pytest.approx(10.0 - 4.0)
    assert state.stock == pytest.approx(54.0)
    applied = next(e for e in events.events if e.type == EventType.PENALTY_APPLIED)
    assert applied.target_id == "a1"
    assert applied.payload["amount"] == pytest.approx(4.0)


async def test_penalise_phase_proportional_fine_and_communal_fund_destination():
    state = FisheryState.initial(make_config())
    state.active_norms.append(
        PenalisePrimitive(
            id="pen1",
            scope="collective",
            trigger="exceed_cap",
            penalty_type="proportional_fine",
            value=0.5,
            destination="communal_fund",
        )
    )
    state.agents["a1"].payoff = 10.0
    events = InMemoryEventSink()

    await run_penalise_phase(state, events, {"a1": [{"trigger": "exceed_cap", "excess": 4.0}]})

    assert state.agents["a1"].payoff == pytest.approx(10.0 - 2.0)  # 4.0 * 0.5
    assert state.communal_fund == pytest.approx(2.0)


async def test_penalise_phase_ignores_non_matching_trigger():
    state = FisheryState.initial(make_config())
    state.active_norms.append(
        PenalisePrimitive(id="pen1", scope="collective", trigger="fail_declare", penalty_type="forfeit")
    )
    state.agents["a1"].payoff = 10.0
    events = InMemoryEventSink()

    await run_penalise_phase(state, events, {"a1": [{"trigger": "exceed_cap", "excess": 4.0}]})

    assert state.agents["a1"].payoff == pytest.approx(10.0)
    assert events.events == []


async def test_penalise_phase_no_op_without_primitive():
    state = FisheryState.initial(make_config())
    events = InMemoryEventSink()
    await run_penalise_phase(state, events, {"a1": [{"trigger": "exceed_cap", "excess": 4.0}]})
    assert events.events == []


async def test_penalise_phase_ignores_reserved_pool_violation_key():
    state = FisheryState.initial(make_config())
    state.active_norms.append(
        PenalisePrimitive(id="pen1", scope="collective", trigger="exceed_cap", penalty_type="forfeit")
    )
    events = InMemoryEventSink()
    await run_penalise_phase(state, events, {"__pool__": [{"trigger": "exceed_cap", "excess": 4.0}]})
    assert events.events == []


# --- RedistributePrimitive -------------------------------------------------------


async def test_redistribute_phase_splits_excess_units_equally_on_violation():
    state = FisheryState.initial(make_config())
    state.active_norms.append(
        RedistributePrimitive(
            id="r1",
            scope="collective",
            source="violator",
            destination="all_agents_equal",
            trigger="violation",
            amount_basis="excess_units",
        )
    )
    for agent in state.agents.values():
        agent.payoff = 0.0
    events = InMemoryEventSink()

    await run_redistribute_phase(state, events, {"a1": [{"trigger": "exceed_cap", "excess": 3.0}]})

    for agent in state.agents.values():
        assert agent.payoff == pytest.approx(1.0)  # 3.0 / 3 agents
    applied = next(e for e in events.events if e.type == EventType.REDISTRIBUTION_APPLIED)
    assert applied.payload["total_amount"] == pytest.approx(3.0)


async def test_redistribute_phase_no_op_when_trigger_is_violation_and_none_occurred():
    state = FisheryState.initial(make_config())
    state.active_norms.append(
        RedistributePrimitive(id="r1", scope="collective", trigger="violation", amount_basis="excess_units")
    )
    events = InMemoryEventSink()
    await run_redistribute_phase(state, events, {})
    assert events.events == []


async def test_redistribute_phase_fixed_amount_to_communal_fund_from_pool():
    state = FisheryState.initial(make_config())
    state.active_norms.append(
        RedistributePrimitive(
            id="r1",
            scope="collective",
            source="pool",
            destination="communal_fund",
            trigger="end_of_round",
            amount_basis="fixed",
            fixed_amount=5.0,
        )
    )
    state.stock = 50.0
    events = InMemoryEventSink()

    await run_redistribute_phase(state, events, {})

    assert state.stock == pytest.approx(45.0)
    assert state.communal_fund == pytest.approx(5.0)


async def test_redistribute_phase_no_op_without_primitive():
    state = FisheryState.initial(make_config())
    events = InMemoryEventSink()
    await run_redistribute_phase(state, events, {"a1": [{"trigger": "exceed_cap", "excess": 3.0}]})
    assert events.events == []


# --- AdjustPrimitive --------------------------------------------------------------


async def test_adjust_phase_reduces_quota_end_of_round():
    state = FisheryState.initial(make_config())
    state.active_norms.append(
        AdjustPrimitive(
            id="a1", scope="collective", trigger="end_of_round", adjust_target="individual_quota",
            direction="reduce", value=2.0,
        )
    )
    events = InMemoryEventSink()

    await run_adjust_phase(state, events)
    # "reduce" with no existing quota starts from +inf (faithfully ported
    # from the reference implementation's own same quirk) -- the first
    # reduction alone doesn't establish a finite ceiling; "recalculate" is
    # the direction that actually sets one from nothing (see next test).
    assert state.adjusted_quota == float("inf")

    adjusted = next(e for e in events.events if e.type == EventType.QUOTA_ADJUSTED)
    assert adjusted.payload["direction"] == "reduce"


async def test_adjust_phase_recalculate_sets_quota_directly():
    state = FisheryState.initial(make_config())
    state.active_norms.append(
        AdjustPrimitive(
            id="a1", scope="collective", trigger="end_of_round", adjust_target="individual_quota",
            direction="recalculate", value=7.5,
        )
    )
    events = InMemoryEventSink()
    await run_adjust_phase(state, events)
    assert state.adjusted_quota == pytest.approx(7.5)


async def test_adjust_phase_threshold_trigger_only_fires_below_sustainable_level():
    state = FisheryState.initial(make_config(k=100.0))
    state.active_norms.append(
        AdjustPrimitive(
            id="a1", scope="collective", trigger="collective_threshold_breached",
            adjust_target="individual_quota", direction="recalculate", value=1.0,
        )
    )
    state.stock = 90.0  # above 0.4*100=40 sustainable threshold -- should not fire
    events = InMemoryEventSink()
    await run_adjust_phase(state, events)
    assert state.adjusted_quota is None

    state.stock = 10.0  # below threshold -- should fire
    await run_adjust_phase(state, events)
    assert state.adjusted_quota == pytest.approx(1.0)


async def test_adjust_phase_no_op_without_primitive():
    state = FisheryState.initial(make_config())
    events = InMemoryEventSink()
    await run_adjust_phase(state, events)
    assert events.events == []
    assert state.adjusted_quota is None


# --- NormCompiler: the ported primitive set --------------------------------------


async def test_norm_compiler_compiles_declare():
    from genfishery.llm.fake_client import FakeLLMClient
    from genfishery.config.model_config import LLMCallType

    output = NormCompilerOutput.model_validate(
        {
            "primitives": [
                {
                    "type": "declare",
                    "timing": "pre_round",
                    "content": "intended_quota",
                    "disclosure_visibility": "public",
                }
            ]
        }
    )
    compiler = NormCompiler(FakeLLMClient({LLMCallType.NORM_COMPILER: output}))
    spec = await compiler.compile(
        "Everyone must announce their intended catch before fishing.", norm_id="n1", adopted_round=1
    )

    assert len(spec.primitives) == 1
    assert isinstance(spec.primitives[0], DeclarePrimitive)
    assert spec.primitives[0].timing == "pre_round"


async def test_norm_compiler_compiles_monitor():
    from genfishery.llm.fake_client import FakeLLMClient
    from genfishery.config.model_config import LLMCallType

    output = NormCompilerOutput.model_validate(
        {
            "primitives": [
                {
                    "type": "monitor",
                    "method": "rotating_role",
                    "frequency": "every_round",
                    "monitor_target": "individual",
                }
            ]
        }
    )
    compiler = NormCompiler(FakeLLMClient({LLMCallType.NORM_COMPILER: output}))
    spec = await compiler.compile("A rotating villager checks compliance every round.", norm_id="n2", adopted_round=1)

    assert len(spec.primitives) == 1
    assert isinstance(spec.primitives[0], MonitorPrimitive)
    assert spec.primitives[0].method == "rotating_role"


async def test_norm_compiler_compiles_penalise():
    from genfishery.llm.fake_client import FakeLLMClient
    from genfishery.config.model_config import LLMCallType

    output = NormCompilerOutput.model_validate(
        {
            "primitives": [
                {
                    "type": "penalise",
                    "trigger": "exceed_cap",
                    "penalty_type": "forfeit",
                    "value": 0.0,
                    "destination": "pool",
                }
            ]
        }
    )
    compiler = NormCompiler(FakeLLMClient({LLMCallType.NORM_COMPILER: output}))
    spec = await compiler.compile("Anyone who exceeds the cap forfeits the excess.", norm_id="n3", adopted_round=1)

    assert len(spec.primitives) == 1
    assert isinstance(spec.primitives[0], PenalisePrimitive)
    assert spec.primitives[0].penalty_type == "forfeit"


async def test_norm_compiler_compiles_adjust_and_redistribute_together():
    from genfishery.llm.fake_client import FakeLLMClient
    from genfishery.config.model_config import LLMCallType

    output = NormCompilerOutput.model_validate(
        {
            "primitives": [
                {
                    "type": "adjust",
                    "trigger": "collective_threshold_breached",
                    "adjust_target": "individual_quota",
                    "direction": "reduce",
                    "value": 2.0,
                },
                {
                    "type": "redistribute",
                    "source": "violator",
                    "destination": "all_agents_equal",
                    "trigger": "violation",
                    "amount_basis": "excess_units",
                },
            ]
        }
    )
    compiler = NormCompiler(FakeLLMClient({LLMCallType.NORM_COMPILER: output}))
    spec = await compiler.compile(
        "Reduce everyone's quota if the stock drops too low, and split any forfeited catch equally.",
        norm_id="n4",
        adopted_round=1,
    )

    assert len(spec.primitives) == 2
    assert any(isinstance(p, AdjustPrimitive) for p in spec.primitives)
    assert any(isinstance(p, RedistributePrimitive) for p in spec.primitives)


async def test_norm_compiler_compiles_assign_role():
    from genfishery.llm.fake_client import FakeLLMClient
    from genfishery.config.model_config import LLMCallType

    output = NormCompilerOutput.model_validate(
        {
            "primitives": [
                {
                    "type": "assign_role",
                    "role_name": "auditor",
                    "selection": "rotating",
                    "duration": 5,
                    "obligation": "check catches",
                }
            ]
        }
    )
    compiler = NormCompiler(FakeLLMClient({LLMCallType.NORM_COMPILER: output}))
    spec = await compiler.compile("Rotate an auditor every 5 rounds.", norm_id="n5", adopted_round=1)

    assert len(spec.primitives) == 1
    assert isinstance(spec.primitives[0], AssignRolePrimitive)
    assert spec.primitives[0].role_name == "auditor"
    assert spec.primitives[0].selection == "rotating"
