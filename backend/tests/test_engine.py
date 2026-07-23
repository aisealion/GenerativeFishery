import pytest

from genfishery.config.fishery_config import FisheryConfig
from genfishery.models.events import EventType
from genfishery.models.norms import CapPrimitive, PenalisePrimitive
from genfishery.sim.engine import (
    check_fishery_collapse,
    run_harvest_phase,
    run_penalise_phase,
    run_simulation,
    run_starvation_check,
    run_strategy_phase,
)
from genfishery.sim.events_sink import InMemoryEventSink
from genfishery.sim.state import FisheryState


class StubDecisionSource:
    """Deterministic decision source for precise arithmetic assertions."""

    def __init__(self, efforts: dict[str, float]):
        self.efforts = efforts

    async def decide_effort(self, agent_id, state):
        return self.efforts[agent_id]


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


def test_default_active_norms_is_empty_every_primitive_is_opt_in():
    state = FisheryState.initial(make_config())
    assert state.active_norms == []
    assert state.round == 0


async def test_harvest_and_regrowth_match_gupta_equations():
    config = make_config(alpha=0.1, r=0.5, k=100.0, initial_stock=100.0, consumption=1.0)
    state = FisheryState.initial(config)
    state.round = 1
    decisions = StubDecisionSource({"a1": 0.5, "a2": 0.5, "a3": 0.5})
    events = InMemoryEventSink()

    await run_strategy_phase(state, decisions, events)
    await run_harvest_phase(state, events)

    # h_i = alpha * e_i * R(t) = 0.1 * 0.5 * 100 = 5 per agent
    for agent in state.agents.values():
        assert agent.last_harvest == pytest.approx(5.0)
        assert agent.payoff == pytest.approx(5.0 - 1.0)  # h_i - consumption

    # R+ = max(0, 100 - 15) = 85
    # R(t+1) = 85 + 0.5*85*(1 - 85/100) = 85 + 0.5*85*0.15 = 85 + 6.375 = 91.375
    assert state.stock == pytest.approx(91.375)

    harvest_events = [e for e in events.events if e.type == EventType.HARVEST_RESOLVED]
    assert len(harvest_events) == 1
    assert harvest_events[0].payload["post_harvest_stock"] == pytest.approx(85.0)


async def test_harvest_never_drives_stock_negative():
    config = make_config(alpha=1.0, r=0.5, k=100.0, initial_stock=10.0, consumption=0.0)
    state = FisheryState.initial(config)
    decisions = StubDecisionSource({"a1": 1.0, "a2": 1.0, "a3": 1.0})
    events = InMemoryEventSink()

    await run_strategy_phase(state, decisions, events)
    await run_harvest_phase(state, events)

    harvest_events = [e for e in events.events if e.type == EventType.HARVEST_RESOLVED]
    assert harvest_events[0].payload["post_harvest_stock"] == 0.0


async def test_harvest_phase_clips_catch_to_an_active_cap_and_records_a_violation():
    config = make_config(alpha=1.0, initial_stock=10.0, consumption=0.0)
    state = FisheryState.initial(config)
    state.active_norms.append(
        CapPrimitive(id="cap1", scope="individual", basis="fixed_units", value=2.0)
    )
    state.agents["a1"].last_effort = 0.5  # requested = 1.0*0.5*10.0 = 5.0
    state.agents["a2"].last_effort = 0.0
    state.agents["a3"].last_effort = 0.0
    events = InMemoryEventSink()

    violations = await run_harvest_phase(state, events)

    assert state.agents["a1"].last_harvest == pytest.approx(2.0)  # clipped, not the requested 5.0
    assert violations["a1"] == [{"trigger": "exceed_cap", "excess": pytest.approx(3.0)}]
    cap_event = next(e for e in events.events if e.type == EventType.CAP_EXCEEDED)
    assert cap_event.actor_id == "a1"
    assert cap_event.payload["observed"] == pytest.approx(5.0)
    assert cap_event.payload["cap_value"] == pytest.approx(2.0)


async def test_penalise_resolves_before_starvation_and_can_cause_it():
    """Exercises the actual round-order guarantee: an automatic penalty from
    Harvest's own cap violation must land before the starvation check, same
    as any other payoff-affecting phase.
    """
    config = make_config(consumption=0.0, alpha=1.0, initial_stock=10.0)
    state = FisheryState.initial(config)
    state.active_norms = [
        CapPrimitive(id="cap1", scope="individual", basis="fixed_units", value=2.0),
        PenalisePrimitive(
            id="pen1", scope="collective", trigger="exceed_cap", penalty_type="forfeit", destination="pool"
        ),
    ]
    state.agents["a1"].last_effort = 0.5  # requested = 5.0, capped to 2.0, excess = 3.0
    state.agents["a2"].last_effort = 0.0
    state.agents["a3"].last_effort = 0.0
    state.agents["a1"].payoff = -1.0  # would go to -2.0 once the forfeit lands

    events = InMemoryEventSink()
    violations = await run_harvest_phase(state, events)
    assert state.agents["a1"].payoff == pytest.approx(-1.0 + 2.0)  # capped harvest applied

    await run_penalise_phase(state, events, violations)
    assert state.agents["a1"].payoff == pytest.approx(1.0 - 3.0)  # excess forfeited

    await run_starvation_check(state, events)
    assert state.agents["a1"].alive is False
    starved = [e for e in events.events if e.type == EventType.AGENT_STARVED]
    assert len(starved) == 1
    assert starved[0].target_id == "a1"
    # a1's payoff was still non-negative right after harvest -- the penalty
    # is what tipped them into starvation, not underharvest.
    assert starved[0].payload["reason"] == "punished"
    assert state.starvation_reasons["a1"] == "punished"


async def test_starvation_from_underharvest_alone_is_labeled_underharvest():
    config = make_config(consumption=5.0)
    state = FisheryState.initial(config)
    state.agents["a1"].payoff = 1.0  # harvest this round won't cover consumption
    state.agents["a1"].last_effort = 0.0
    state.agents["a2"].last_effort = 0.0
    state.agents["a3"].last_effort = 0.0
    events = InMemoryEventSink()

    await run_harvest_phase(state, events)  # 0 harvest - 5 consumption == -5.0 delta
    assert state.agents["a1"].payoff == pytest.approx(1.0 - 5.0)

    await run_starvation_check(state, events)
    assert state.agents["a1"].alive is False
    starved = next(e for e in events.events if e.type == EventType.AGENT_STARVED)
    assert starved.payload["reason"] == "underharvest"
    assert state.starvation_reasons["a1"] == "underharvest"


def test_collapse_on_stock_floor():
    config = make_config(r_min=10.0, n_min=1)
    state = FisheryState.initial(config)
    state.stock = 5.0
    assert check_fishery_collapse(state) is True


def test_collapse_on_population_floor():
    config = make_config(r_min=0.0, n_min=3)
    state = FisheryState.initial(config)
    state.agents["a1"].alive = False
    assert check_fishery_collapse(state) is True


async def test_run_simulation_stops_at_max_rounds_if_no_collapse():
    config = make_config(alpha=0.0, consumption=0.0, max_rounds=5)  # no harvest -> never collapses
    state = FisheryState.initial(config)
    decisions = StubDecisionSource({"a1": 0.0, "a2": 0.0, "a3": 0.0})
    events = InMemoryEventSink()

    survival_time = await run_simulation(state, decisions, events)
    assert survival_time == 5
    assert state.collapsed is False


async def test_run_simulation_reports_survival_time_at_collapse():
    config = make_config(alpha=1.0, r=0.0, initial_stock=10.0, consumption=0.0, n_min=3, r_min=0.0)
    state = FisheryState.initial(config)
    # Max effort drains the stock to 0 in one round -> collapse at round 1.
    decisions = StubDecisionSource({"a1": 1.0, "a2": 1.0, "a3": 1.0})
    events = InMemoryEventSink()

    survival_time = await run_simulation(state, decisions, events)
    assert survival_time == 1
    assert state.collapsed is True
    collapse_event = next(e for e in events.events if e.type == EventType.FISHERY_COLLAPSED)
    assert collapse_event.payload["reasons"] == ["resource_depletion"]


async def test_collapse_event_records_population_loss_reason():
    config = make_config(consumption=5.0, r_min=0.0, n_min=3)  # underharvest starves everyone
    state = FisheryState.initial(config)
    decisions = StubDecisionSource({"a1": 0.0, "a2": 0.0, "a3": 0.0})
    events = InMemoryEventSink()

    await run_simulation(state, decisions, events)

    collapse_event = next(e for e in events.events if e.type == EventType.FISHERY_COLLAPSED)
    # Population dropped below n_min *and* the deaths were underharvest-caused.
    assert set(collapse_event.payload["reasons"]) == {"population_loss", "underharvest_death"}


async def test_collapse_event_records_all_three_reasons_when_all_apply():
    config = make_config(
        alpha=0.1, r=0.0, initial_stock=10.0, consumption=1.5, r_min=7.0, n_min=3
    )
    state = FisheryState.initial(config)
    # h_i = 0.1*1.0*10.0 = 1.0 per agent -- covers r_min (stock ends at 7.0,
    # <= r_min -> resource_depletion) but not the 1.5 consumption cost, so
    # every agent's payoff goes negative too -> population_loss AND
    # underharvest_death, same round.
    decisions = StubDecisionSource({"a1": 1.0, "a2": 1.0, "a3": 1.0})
    events = InMemoryEventSink()

    await run_simulation(state, decisions, events)

    collapse_event = next(e for e in events.events if e.type == EventType.FISHERY_COLLAPSED)
    assert set(collapse_event.payload["reasons"]) == {
        "resource_depletion",
        "population_loss",
        "underharvest_death",
    }


async def test_single_underharvest_death_collapses_even_with_population_and_stock_healthy():
    """The core new behavior: collapse is triggered the moment *any* agent
    starves from underharvest, independent of n_min/r_min headroom."""
    config = make_config(consumption=100.0, r_min=0.0, n_min=1, initial_stock=10_000.0, k=10_000.0)
    state = FisheryState.initial(config)
    # a1 fishes 0 effort and starves from underharvest; a2/a3 fish hard enough
    # to stay alive and the stock barely moves -- population (2 of 3 alive)
    # and stock are both nowhere near their floors.
    decisions = StubDecisionSource({"a1": 0.0, "a2": 1.0, "a3": 1.0})
    events = InMemoryEventSink()

    await run_simulation(state, decisions, events)

    assert state.round == 1
    collapse_event = next(e for e in events.events if e.type == EventType.FISHERY_COLLAPSED)
    assert collapse_event.payload["reasons"] == ["underharvest_death"]
    assert collapse_event.payload["n_alive"] == 2  # a2/a3 still alive -- not a population-floor collapse


async def test_penalise_caused_death_does_not_trigger_underharvest_collapse():
    """Explicit project requirement: a penalty-caused starvation must not, by
    itself, collapse the fishery -- only underharvest deaths do."""
    config = make_config(consumption=0.0, alpha=1.0, initial_stock=10.0, r_min=0.0, n_min=1)
    state = FisheryState.initial(config)
    state.active_norms = [
        CapPrimitive(id="cap1", scope="individual", basis="fixed_units", value=2.0),
        PenalisePrimitive(
            id="pen1", scope="collective", trigger="exceed_cap", penalty_type="forfeit", destination="pool"
        ),
    ]
    state.agents["a1"].last_effort = 0.5  # requested = 5.0, capped to 2.0, excess = 3.0
    state.agents["a2"].last_effort = 0.0
    state.agents["a3"].last_effort = 0.0
    state.agents["a1"].payoff = 0.5  # survives harvest alone (0.5+2.0=2.5), not the forfeit

    events = InMemoryEventSink()
    violations = await run_harvest_phase(state, events)
    await run_penalise_phase(state, events, violations)
    had_underharvest_death = await run_starvation_check(state, events)

    assert state.agents["a1"].alive is False
    assert state.starvation_reasons["a1"] == "punished"
    assert had_underharvest_death is False
    assert check_fishery_collapse(state, underharvest_death_this_round=had_underharvest_death) is False
