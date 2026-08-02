from genfishery.config.fishery_config import FisheryConfig
from genfishery.config.model_config import LLMCallType, ModelConfig
from genfishery.llm.fake_client import FakeLLMClient
from genfishery.models.events import EventType
from genfishery.sim.decisions import PolicyProposal, ProposalDecision
from genfishery.sim.engine import run_norm_adoption, run_propose_phase, run_vote_phase
from genfishery.sim.events_sink import InMemoryEventSink
from genfishery.sim.policies import LLMDecisionSource
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


def make_decisions(script: dict) -> LLMDecisionSource:
    return LLMDecisionSource(FakeLLMClient(script), ModelConfig.default())


async def test_propose_phase_updates_personal_norms_and_logs_events():
    state = FisheryState.initial(make_config())
    proposals_by_agent = {
        "a1": ProposalDecision(personal_norm="fish less", community_proposal="cap effort at 0.3"),
        "a2": ProposalDecision(personal_norm="fish less too", community_proposal="cap effort at 0.3"),
        "a3": ProposalDecision(personal_norm="fish more", community_proposal="no cap needed"),
    }

    class ScriptedProposalSource:
        async def decide_proposal(self, agent_id, state):
            return proposals_by_agent[agent_id]

    events = InMemoryEventSink()
    # No SE agent wired up -- operationalization is left blank rather than
    # discussed (see `run_propose_phase`'s docstring).
    proposals = await run_propose_phase(state, ScriptedProposalSource(), events)

    assert proposals == {
        "a1": PolicyProposal(community_proposal="cap effort at 0.3", operationalization=""),
        "a2": PolicyProposal(community_proposal="cap effort at 0.3", operationalization=""),
        "a3": PolicyProposal(community_proposal="no cap needed", operationalization=""),
    }
    assert state.agent_norms["a1"] == "fish less"
    assert state.agent_norms["a3"] == "fish more"

    personal_events = [e for e in events.events if e.type == EventType.PERSONAL_NORM_UPDATED]
    proposal_events = [e for e in events.events if e.type == EventType.PROPOSAL_MADE]
    assert len(personal_events) == 3
    assert all(e.visibility.value == "actor_only" for e in personal_events)
    assert len(proposal_events) == 3
    assert all(e.visibility.value == "public" for e in proposal_events)
    assert proposal_events[0].payload["operationalization"] == ""


async def test_vote_phase_dedupes_candidates_and_tallies_majority():
    state = FisheryState.initial(make_config())
    cap = PolicyProposal(community_proposal="cap at 0.3", operationalization="check weekly")
    no_cap = PolicyProposal(community_proposal="no cap", operationalization="n/a")
    proposals = {"a1": cap, "a2": cap, "a3": no_cap}
    votes = {"a1": cap, "a2": no_cap, "a3": cap}

    class ScriptedVoteSource:
        async def decide_vote(self, agent_id, state, *, candidates):
            assert set(candidates) == {cap, no_cap}
            return votes[agent_id]

    events = InMemoryEventSink()
    winner = await run_vote_phase(state, ScriptedVoteSource(), events, proposals)

    assert winner == cap
    result_event = next(e for e in events.events if e.type == EventType.VOTE_RESULT)
    assert result_event.payload["tally"] == {"cap at 0.3": 2, "no cap": 1}
    assert result_event.payload["operationalization"] == "check weekly"
    cast_events = [e for e in events.events if e.type == EventType.VOTE_CAST]
    assert len(cast_events) == 3
    assert all(e.visibility.value == "actor_only" for e in cast_events)


async def test_vote_phase_returns_none_when_no_proposals():
    state = FisheryState.initial(make_config())

    class NeverCalledSource:
        async def decide_vote(self, agent_id, state, *, candidates):
            raise AssertionError("should not be called with no candidates")

    events = InMemoryEventSink()
    winner = await run_vote_phase(state, NeverCalledSource(), events, {})
    assert winner is None
    assert events.events == []


async def test_run_norm_adoption_updates_group_norm_text_on_a_new_winner():
    state = FisheryState.initial(make_config())
    winning_text = "No one should catch more than 20 units."
    fake_llm = FakeLLMClient(
        {
            LLMCallType.PROPOSAL: ProposalDecision(personal_norm="ok", community_proposal=winning_text),
            LLMCallType.VOTE: lambda response_model, system, prompt: response_model(chosen_id="1"),
        }
    )
    decisions = LLMDecisionSource(fake_llm, ModelConfig.default())
    events = InMemoryEventSink()

    await run_norm_adoption(state, decisions, events)

    assert state.group_norm_text == winning_text
    adopted_event = next(e for e in events.events if e.type == EventType.NORM_ADOPTED)
    assert adopted_event.payload["community_proposal"] == winning_text


async def test_run_norm_adoption_is_a_no_op_when_the_winner_matches_the_current_norm():
    state = FisheryState.initial(make_config())
    state.group_norm_text = "Already the policy."
    fake_llm = FakeLLMClient(
        {
            LLMCallType.PROPOSAL: ProposalDecision(
                personal_norm="ok", community_proposal="Already the policy."
            ),
            LLMCallType.VOTE: lambda response_model, system, prompt: response_model(chosen_id="1"),
        }
    )
    decisions = LLMDecisionSource(fake_llm, ModelConfig.default())
    events = InMemoryEventSink()

    await run_norm_adoption(state, decisions, events)

    assert not any(e.type == EventType.NORM_ADOPTED for e in events.events)
