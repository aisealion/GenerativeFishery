"""Tests for the fishery-councillor operationalization discussion: after a
proposal, the proposing agent and the councillor go back and forth for a
configured number of turn-pairs, and the final turn's agent reply becomes the
finalized operationalization text (see `sim.engine.run_operationalization_discussion_phase`).
"""

from genfishery.config.fishery_config import FisheryConfig
from genfishery.models.events import EventType, Visibility
from genfishery.sim.decisions import PolicyProposal
from genfishery.sim.engine import run_operationalization_discussion_phase, run_propose_phase
from genfishery.sim.events_sink import InMemoryEventSink
from genfishery.sim.state import FisheryState


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


class FakeCouncillorClient:
    def __init__(self) -> None:
        self.sessions_started: list[str] = []
        self.asks: list[tuple[str, str]] = []
        self._next_session_id = 0

    async def start_session(self, title: str) -> str:
        self.sessions_started.append(title)
        self._next_session_id += 1
        return f"session-{self._next_session_id}"

    async def ask(self, session_id: str, message: str) -> str:
        self.asks.append((session_id, message))
        return f"councillor question about: {message[:20]}"


class ScriptedGovernanceSource:
    """Only implements what `run_propose_phase`/`run_operationalization_discussion_phase`
    actually call -- `decide_proposal` and `decide_councillor_reply`.
    """

    def __init__(self, *, community_proposal: str) -> None:
        self._community_proposal = community_proposal
        self.reply_calls: list[dict] = []

    async def decide_proposal(self, agent_id, state):
        from genfishery.sim.decisions import ProposalDecision

        return ProposalDecision(personal_norm="ok", community_proposal=self._community_proposal)

    async def decide_councillor_reply(
        self, agent_id, state, *, community_proposal, transcript, councillor_message, is_final_turn
    ):
        self.reply_calls.append(
            {
                "agent_id": agent_id,
                "community_proposal": community_proposal,
                "transcript_len": len(transcript),
                "councillor_message": councillor_message,
                "is_final_turn": is_final_turn,
            }
        )
        return "FINALIZED: cap effort and check weekly" if is_final_turn else "still discussing"


async def test_discussion_phase_runs_configured_rounds_and_returns_final_reply():
    state = FisheryState.initial(make_config())
    councillor = FakeCouncillorClient()
    decisions = ScriptedGovernanceSource(community_proposal="cap effort at 0.3")
    events = InMemoryEventSink()

    result = await run_operationalization_discussion_phase(
        state,
        decisions,
        events,
        councillor,
        agent_id="a1",
        community_proposal="cap effort at 0.3",
        rounds=3,
    )

    assert result == "FINALIZED: cap effort and check weekly"
    assert len(councillor.sessions_started) == 1
    # 1 opening ask + (rounds - 1) follow-up asks, since the final turn's
    # reply is finalized rather than relayed back to the councillor.
    assert len(councillor.asks) == 3
    assert len(decisions.reply_calls) == 3
    assert [c["is_final_turn"] for c in decisions.reply_calls] == [False, False, True]
    # Transcript grows by one (councillor, agent) pair each turn.
    assert [c["transcript_len"] for c in decisions.reply_calls] == [0, 2, 4]


async def test_discussion_phase_records_actor_only_events_per_turn():
    state = FisheryState.initial(make_config())
    councillor = FakeCouncillorClient()
    decisions = ScriptedGovernanceSource(community_proposal="cap effort at 0.3")
    events = InMemoryEventSink()

    await run_operationalization_discussion_phase(
        state,
        decisions,
        events,
        councillor,
        agent_id="a1",
        community_proposal="cap effort at 0.3",
        rounds=2,
    )

    questions = [e for e in events.events if e.type == EventType.COUNCILLOR_QUESTION]
    replies = [e for e in events.events if e.type == EventType.COUNCILLOR_DISCUSSION_REPLY]
    assert len(questions) == 2
    assert len(replies) == 2
    for event in questions + replies:
        assert event.actor_id == "a1"
        assert event.visibility == Visibility.ACTOR_ONLY


async def test_propose_phase_uses_councillor_to_finalize_operationalization():
    state = FisheryState.initial(
        make_config(initial_agent_ids=["a1"], operationalization_discussion_rounds=2)
    )
    councillor = FakeCouncillorClient()
    decisions = ScriptedGovernanceSource(community_proposal="cap effort at 0.3")
    events = InMemoryEventSink()

    proposals = await run_propose_phase(state, decisions, events, councillor=councillor)

    assert proposals == {
        "a1": PolicyProposal(
            community_proposal="cap effort at 0.3", operationalization="FINALIZED: cap effort and check weekly"
        )
    }


async def test_propose_phase_leaves_operationalization_blank_without_councillor():
    state = FisheryState.initial(make_config(initial_agent_ids=["a1"]))
    decisions = ScriptedGovernanceSource(community_proposal="cap effort at 0.3")
    events = InMemoryEventSink()

    proposals = await run_propose_phase(state, decisions, events)

    assert proposals == {"a1": PolicyProposal(community_proposal="cap effort at 0.3", operationalization="")}
