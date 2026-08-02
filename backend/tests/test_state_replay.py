"""Round-trip tests for `sim.state_replay.reconstruct_state`: run a fishery
for a few rounds, reconstruct its state purely from the event log, and
confirm the reconstruction matches the live state field-for-field.
"""

from genfishery.config.fishery_config import FisheryConfig
from genfishery.models.events import Event, EventType
from genfishery.sim.decisions import ProposalDecision
from genfishery.sim.engine import run_round
from genfishery.sim.events_sink import InMemoryEventSink
from genfishery.sim.state import FisheryState
from genfishery.sim.state_replay import reconstruct_state


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
        n_min=1,
        max_rounds=None,
    )
    defaults.update(overrides)
    return FisheryConfig(**defaults)


class StubGovernanceSource:
    """Deterministic decisions covering strategy/propose/vote, so
    HARVEST_RESOLVED, PERSONAL_NORM_UPDATED, and NORM_ADOPTED all get
    exercised in the same run.
    """

    def __init__(self, *, community_proposal: str) -> None:
        self._community_proposal = community_proposal

    async def decide_effort(self, agent_id, state):
        return 0.3

    async def decide_proposal(self, agent_id, state):
        return ProposalDecision(personal_norm=f"{agent_id}-norm", community_proposal=self._community_proposal)

    async def decide_vote(self, agent_id, state, *, candidates):
        return candidates[0]


async def test_reconstruct_state_returns_none_with_no_history():
    events = InMemoryEventSink()
    result = await reconstruct_state("test", events, make_config())
    assert result is None


async def test_reconstruct_state_matches_live_state_after_several_rounds():
    config = make_config()
    state = FisheryState.initial(config)
    events = InMemoryEventSink()

    # Mirrors FisheryRunner._bootstrap's one-time init event for a fresh
    # fishery -- reconstruct_state needs it to recover persona assignment,
    # which is otherwise randomized and unrecoverable from later events alone.
    await events.record(
        Event.create(
            fishery_id=config.fishery_id,
            round=0,
            phase="init",
            type=EventType.FISHERY_INITIALIZED,
            payload={
                "persona_descriptions": state.persona_descriptions,
                "persona_types": state.persona_types,
            },
        )
    )

    decisions = StubGovernanceSource(community_proposal="Fish moderately.")
    for _ in range(3):
        await run_round(state, decisions, events, enable_governance=True)

    reconstructed = await reconstruct_state(config.fishery_id, events, config)
    assert reconstructed is not None

    assert reconstructed.round == state.round
    assert reconstructed.stock == state.stock
    assert reconstructed.group_norm_text == state.group_norm_text
    assert reconstructed.agent_norms == state.agent_norms
    assert reconstructed.persona_descriptions == state.persona_descriptions
    assert reconstructed.persona_types == state.persona_types
    assert reconstructed.starvation_reasons == state.starvation_reasons

    assert set(reconstructed.agents) == set(state.agents)
    for agent_id, live_agent in state.agents.items():
        replayed_agent = reconstructed.agents[agent_id]
        assert replayed_agent.alive == live_agent.alive
        assert replayed_agent.last_effort == live_agent.last_effort
        assert replayed_agent.last_harvest == live_agent.last_harvest
        assert replayed_agent.payoff == live_agent.payoff


async def test_reconstruct_state_replays_starvation_and_collapse():
    config = make_config(consumption=1000.0, r_min=0.0, n_min=1)  # guarantees underharvest death -> collapse
    state = FisheryState.initial(config)
    events = InMemoryEventSink()
    await events.record(
        Event.create(
            fishery_id=config.fishery_id,
            round=0,
            phase="init",
            type=EventType.FISHERY_INITIALIZED,
            payload={
                "persona_descriptions": state.persona_descriptions,
                "persona_types": state.persona_types,
            },
        )
    )

    class NoGovernance:
        async def decide_effort(self, agent_id, state):
            return 0.0

    await run_round(state, NoGovernance(), events, enable_governance=False)
    assert state.collapsed is True

    reconstructed = await reconstruct_state(config.fishery_id, events, config)
    assert reconstructed is not None
    assert reconstructed.collapsed is True
    assert reconstructed.collapse_reasons == state.collapse_reasons
    for agent_id, live_agent in state.agents.items():
        assert reconstructed.agents[agent_id].alive == live_agent.alive
    assert reconstructed.starvation_reasons == state.starvation_reasons
