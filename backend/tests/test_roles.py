from genfishery.config.fishery_config import FisheryConfig
from genfishery.config.model_config import LLMCallType, ModelConfig
from genfishery.llm.fake_client import FakeLLMClient
from genfishery.models.events import EventType
from genfishery.models.norms import AssignRolePrimitive
from genfishery.sim.decisions import NominationDecision, PolicyProposal, ProposalDecision, proposal_candidate_key
from genfishery.sim.engine import (
    _resolve_assign_role,
    run_election_phase,
    run_norm_adoption,
    update_role_rotations,
)
from genfishery.sim.events_sink import InMemoryEventSink
from genfishery.sim.norm_compiler import NormCompiler
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
        initial_agent_ids=["a1", "a2", "a3", "a4"],
        r_min=0.0,
        n_min=2,
        max_rounds=None,
    )
    defaults.update(overrides)
    return FisheryConfig(**defaults)


async def test_election_with_single_nominee_skips_vote_call():
    state = FisheryState.initial(make_config())
    nominate = {"a1": True, "a2": False, "a3": False, "a4": False}

    class Source:
        async def decide_nomination(self, agent_id, state, *, role_name):
            return nominate[agent_id]

        async def decide_election_vote(self, agent_id, state, *, role_name, candidates):
            raise AssertionError("should not be called with only one nominee")

    events = InMemoryEventSink()
    winner = await run_election_phase(state, Source(), events, role_name="monitor")

    assert winner == "a1"
    assert state.roles["monitor"] == "a1"
    elected = next(e for e in events.events if e.type == EventType.ROLE_ELECTED)
    assert elected.payload == {"role_name": "monitor", "agent_id": "a1", "selection": "elected"}


async def test_election_with_multiple_nominees_tallies_votes():
    state = FisheryState.initial(make_config())
    nominate = {"a1": True, "a2": True, "a3": False, "a4": False}
    votes = {"a1": "a2", "a2": "a2", "a3": "a1", "a4": "a2"}

    class Source:
        async def decide_nomination(self, agent_id, state, *, role_name):
            return nominate[agent_id]

        async def decide_election_vote(self, agent_id, state, *, role_name, candidates):
            assert set(candidates) == {"a1", "a2"}
            return votes[agent_id]

    events = InMemoryEventSink()
    winner = await run_election_phase(state, Source(), events, role_name="monitor")
    assert winner == "a2"
    assert state.roles["monitor"] == "a2"


async def test_election_with_no_nominees_leaves_role_vacant():
    state = FisheryState.initial(make_config())

    class Source:
        async def decide_nomination(self, agent_id, state, *, role_name):
            return False

        async def decide_election_vote(self, agent_id, state, *, role_name, candidates):
            raise AssertionError("no nominees means no vote")

    events = InMemoryEventSink()
    winner = await run_election_phase(state, Source(), events, role_name="monitor")

    assert winner is None
    assert "monitor" not in state.roles
    called_event = next(e for e in events.events if e.type == EventType.ROLE_ELECTION_CALLED)
    assert called_event.payload == {"role_name": "monitor", "nominees": []}
    assert not any(e.type == EventType.ROLE_ELECTED for e in events.events)


async def test_rotation_cycles_through_roster_by_duration_and_skips_dead_agents():
    state = FisheryState.initial(make_config())
    state.active_norms.append(
        AssignRolePrimitive(id="r1", scope="collective", role_name="monitor", selection="rotating", duration=2)
    )

    events = InMemoryEventSink()
    state.round = 0
    await update_role_rotations(state, events)
    assert state.roles["monitor"] == "a1"

    state.round = 1  # still within the first 2-round term
    await update_role_rotations(state, events)
    assert state.roles["monitor"] == "a1"

    state.round = 2  # term elapsed -> advance to next in roster
    await update_role_rotations(state, events)
    assert state.roles["monitor"] == "a2"

    # a2 dies -> rotation should skip them going forward. Roster is now
    # [a1, a3, a4]; elapsed terms = (4-0)//2 = 2 -> index 2 % 3 -> "a4".
    state.agents["a2"].alive = False
    state.round = 4
    await update_role_rotations(state, events)
    assert state.roles["monitor"] == "a4"


async def test_random_selection_reproducibly_picks_within_each_window():
    state = FisheryState.initial(make_config())
    state.active_norms.append(
        AssignRolePrimitive(id="r1", scope="collective", role_name="auditor", selection="random", duration=3)
    )
    events = InMemoryEventSink()

    state.round = 0
    await update_role_rotations(state, events)
    first_window_holder = state.roles["auditor"]

    state.round = 1  # still within the first 3-round window -- same holder
    await update_role_rotations(state, events)
    assert state.roles["auditor"] == first_window_holder

    state.round = 3  # next window -- re-rolled (possibly, but deterministically)
    await update_role_rotations(state, events)
    assert state.roles["auditor"] in ["a1", "a2", "a3", "a4"]

    # Replaying the same call sequence from scratch (so the rotation schedule
    # starts at the same round=0 baseline) reproduces the same holder.
    state2 = FisheryState.initial(make_config())
    state2.active_norms.append(
        AssignRolePrimitive(id="r1", scope="collective", role_name="auditor", selection="random", duration=3)
    )
    events2 = InMemoryEventSink()
    state2.round = 0
    await update_role_rotations(state2, events2)
    state2.round = 3
    await update_role_rotations(state2, events2)
    assert state2.roles["auditor"] == state.roles["auditor"]


async def test_resolve_assign_role_elected_runs_the_election_phase():
    state = FisheryState.initial(make_config())
    grant = AssignRolePrimitive(id="r1", scope="collective", role_name="monitor", selection="elected")

    class Source:
        async def decide_nomination(self, agent_id, state, *, role_name):
            return agent_id == "a2"

    events = InMemoryEventSink()
    await _resolve_assign_role(state, Source(), grant, events)
    assert state.roles["monitor"] == "a2"


async def test_resolve_assign_role_rotating_only_starts_the_schedule():
    """Actual holder assignment happens uniformly in `update_role_rotations`,
    not here -- this just marks when the rotation schedule started."""
    state = FisheryState.initial(make_config())
    state.round = 5
    grant = AssignRolePrimitive(id="r1", scope="collective", role_name="monitor", selection="rotating", duration=2)

    class NeverCalledSource:
        async def decide_nomination(self, agent_id, state, *, role_name):
            raise AssertionError("rotating selection must not trigger nomination")

    events = InMemoryEventSink()
    await _resolve_assign_role(state, NeverCalledSource(), grant, events)
    assert state.role_rotation_start["monitor"] == 5
    assert "monitor" not in state.roles  # not assigned yet -- update_role_rotations does that


async def test_norm_adoption_compiles_and_resolves_an_elected_assign_role():
    from genfishery.sim.norm_compiler import NormCompilerOutput

    state = FisheryState.initial(make_config())
    winning_text = "The community should elect a monitor to oversee catches."
    winning_proposal = PolicyProposal(community_proposal=winning_text, operationalization="Hold monthly reviews.")
    fake_llm = FakeLLMClient(
        {
            LLMCallType.PROPOSAL: ProposalDecision(
                personal_norm="ok", community_proposal=winning_text, operationalization="Hold monthly reviews."
            ),
            LLMCallType.VOTE: lambda response_model, system, prompt: response_model(
                chosen_text=proposal_candidate_key(winning_proposal)
            ),
            LLMCallType.NORM_COMPILER: NormCompilerOutput.model_validate(
                {
                    "primitives": [
                        {
                            "type": "assign_role",
                            "role_name": "monitor",
                            "selection": "elected",
                            "duration": 5,
                            "obligation": "review catches",
                        }
                    ]
                }
            ),
            # Only the first agent (roster order) self-nominates, so the
            # election resolves via the single-nominee path -- avoids also
            # needing to script the (differently-shaped) election-vote call.
            LLMCallType.ELECTION_DECISION: [
                NominationDecision(self_nominate=True),
                NominationDecision(self_nominate=False),
                NominationDecision(self_nominate=False),
                NominationDecision(self_nominate=False),
            ],
        }
    )
    decisions = LLMDecisionSource(fake_llm, ModelConfig.default())
    compiler = NormCompiler(fake_llm)
    events = InMemoryEventSink()

    await run_norm_adoption(state, decisions, compiler, events)

    assert any(isinstance(p, AssignRolePrimitive) and p.role_name == "monitor" for p in state.active_norms)
    assert state.roles["monitor"] == "a1"
