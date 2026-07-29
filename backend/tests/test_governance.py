from genfishery.config.fishery_config import FisheryConfig
from genfishery.config.model_config import LLMCallType, ModelConfig
from genfishery.llm.fake_client import FakeLLMClient
from genfishery.models.events import EventType
from genfishery.models.norms import CapPrimitive, PeerObservability, PenalisePrimitive
from genfishery.sim.decisions import PolicyProposal, ProposalDecision
from genfishery.sim.engine import run_norm_adoption, run_propose_phase, run_vote_phase
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
        "a1": ProposalDecision(
            personal_norm="fish less", community_proposal="cap effort at 0.3", operationalization="check weekly"
        ),
        "a2": ProposalDecision(
            personal_norm="fish less too",
            community_proposal="cap effort at 0.3",
            operationalization="check weekly",
        ),
        "a3": ProposalDecision(
            personal_norm="fish more", community_proposal="no cap needed", operationalization="n/a"
        ),
    }

    class ScriptedProposalSource:
        async def decide_proposal(self, agent_id, state):
            return proposals_by_agent[agent_id]

    events = InMemoryEventSink()
    proposals = await run_propose_phase(state, ScriptedProposalSource(), events)

    assert proposals == {
        "a1": PolicyProposal(community_proposal="cap effort at 0.3", operationalization="check weekly"),
        "a2": PolicyProposal(community_proposal="cap effort at 0.3", operationalization="check weekly"),
        "a3": PolicyProposal(community_proposal="no cap needed", operationalization="n/a"),
    }
    assert state.agent_norms["a1"] == "fish less"
    assert state.agent_norms["a3"] == "fish more"

    personal_events = [e for e in events.events if e.type == EventType.PERSONAL_NORM_UPDATED]
    proposal_events = [e for e in events.events if e.type == EventType.PROPOSAL_MADE]
    assert len(personal_events) == 3
    assert all(e.visibility.value == "actor_only" for e in personal_events)
    assert len(proposal_events) == 3
    assert all(e.visibility.value == "public" for e in proposal_events)
    assert proposal_events[0].payload["operationalization"] == "check weekly"


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


async def test_norm_compiler_compiles_cap():
    from genfishery.sim.norm_compiler import NormCompilerOutput

    output = NormCompilerOutput.model_validate(
        {
            "primitives": [
                {
                    "type": "cap",
                    "scope": "individual",
                    "basis": "fixed_units",
                    "value": 20.0,
                    "cap_scope": "per_agent_per_round",
                }
            ]
        }
    )
    llm = FakeLLMClient({LLMCallType.NORM_COMPILER: output})
    compiler = NormCompiler(llm)

    spec = await compiler.compile(
        "No one should catch more than 20 units.", norm_id="norm_r1", adopted_round=1
    )

    assert len(spec.primitives) == 1
    rule = spec.primitives[0]
    assert isinstance(rule, CapPrimitive)
    assert rule.basis == "fixed_units"
    assert rule.value == 20.0
    assert rule.id == "norm_r1_p0"


async def test_norm_compiler_compiles_penalise():
    from genfishery.sim.norm_compiler import NormCompilerOutput

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
    llm = FakeLLMClient({LLMCallType.NORM_COMPILER: output})
    compiler = NormCompiler(llm)

    spec = await compiler.compile(
        "Anyone who exceeds their cap forfeits the excess.",
        norm_id="norm_r2",
        adopted_round=2,
    )

    assert len(spec.primitives) == 1
    penalty = spec.primitives[0]
    assert isinstance(penalty, PenalisePrimitive)
    assert penalty.penalty_type == "forfeit"
    assert penalty.destination == "pool"


async def test_norm_compiler_could_not_compile_returns_empty_primitives():
    from genfishery.sim.norm_compiler import NormCompilerOutput

    llm = FakeLLMClient({LLMCallType.NORM_COMPILER: NormCompilerOutput(primitives=[])})
    compiler = NormCompiler(llm)

    spec = await compiler.compile("Everyone should be nice to each other.", norm_id="norm_r3", adopted_round=3)
    assert spec.primitives == []


async def test_norm_compiler_caches_by_raw_text_hash():
    from genfishery.sim.norm_compiler import NormCompilerOutput

    output = NormCompilerOutput(primitives=[])
    llm = FakeLLMClient({LLMCallType.NORM_COMPILER: [output]})  # queue of exactly 1
    compiler = NormCompiler(llm)

    first = await compiler.compile("same text", norm_id="norm_a", adopted_round=1)
    # A second compile of identical text must hit the cache, not the (now
    # exhausted) queue -- if it re-invoked the LLM this would raise.
    second = await compiler.compile("same text", norm_id="norm_b", adopted_round=2)
    assert first is second


async def test_run_norm_adoption_updates_group_norm_and_active_norms_on_success():
    from genfishery.sim.norm_compiler import NormCompilerOutput

    state = FisheryState.initial(make_config())
    winning_text = "No one should catch more than 20 units."
    script = {
        LLMCallType.PROPOSAL: ProposalDecision(
            personal_norm="ok", community_proposal=winning_text, operationalization="Review weekly."
        ),
        LLMCallType.NORM_COMPILER: NormCompilerOutput.model_validate(
            {
                "primitives": [
                    {
                        "type": "cap",
                        "scope": "individual",
                        "basis": "fixed_units",
                        "value": 20.0,
                    }
                ]
            }
        ),
    }
    script[LLMCallType.VOTE] = lambda response_model, system, prompt: response_model(
        chosen_id="1"
    )
    fake_llm = FakeLLMClient(script)
    decisions = LLMDecisionSource(fake_llm, ModelConfig.default())

    compiler = NormCompiler(fake_llm)
    events = InMemoryEventSink()

    await run_norm_adoption(state, decisions, compiler, events)

    assert state.group_norm_text == winning_text
    assert any(isinstance(p, CapPrimitive) and p.value == 20.0 for p in state.active_norms)
    assert any(e.type == EventType.NORM_ADOPTED for e in events.events)


async def test_run_norm_adoption_could_not_compile_still_updates_group_norm_text():
    from genfishery.sim.norm_compiler import NormCompilerOutput

    state = FisheryState.initial(make_config())
    winning_text = "Be kind to your neighbors."
    fake_llm = FakeLLMClient(
        {
            LLMCallType.PROPOSAL: ProposalDecision(
                personal_norm="ok", community_proposal=winning_text, operationalization="Just be nice."
            ),
            LLMCallType.NORM_COMPILER: NormCompilerOutput(primitives=[]),
            LLMCallType.VOTE: lambda response_model, system, prompt: response_model(
                chosen_id="1"
            ),
        }
    )
    decisions = LLMDecisionSource(fake_llm, ModelConfig.default())

    original_norms = list(state.active_norms)
    compiler = NormCompiler(fake_llm)
    events = InMemoryEventSink()

    await run_norm_adoption(state, decisions, compiler, events)

    assert state.group_norm_text == winning_text  # descriptive belief still updates
    assert state.active_norms == original_norms  # nothing mechanically enforced
    assert any(e.type == EventType.NORM_COULD_NOT_COMPILE for e in events.events)


async def test_peer_observability_is_silent_until_a_norm_enables_it_then_activates():
    """Effort/payoff visibility across villagers is opt-in, same pattern as
    every other primitive: nobody's prompt shows another villager's
    effort/payoff until a community vote is compiled into a PeerObservability
    primitive.
    """
    from genfishery.sim.decisions import EffortDecision
    from genfishery.sim.norm_compiler import NormCompilerOutput

    state = FisheryState.initial(make_config())
    for agent in state.agents.values():
        agent.last_effort = 0.3

    fake_llm = FakeLLMClient({LLMCallType.EFFORT_DECISION: EffortDecision(effort=0.3)})
    decisions = LLMDecisionSource(fake_llm, ModelConfig.default())
    await decisions.decide_effort("a2", state)
    prompt = fake_llm.calls[-1][2]
    # The roster still names a1 (identity/alive-status is always visible),
    # but not their effort/payoff numbers.
    assert "a1: active" in prompt
    assert "a1: effort=" not in prompt

    winning_text = "Let's make everyone's effort and earnings visible to the whole community."
    script = {
        LLMCallType.EFFORT_DECISION: EffortDecision(effort=0.3),
        LLMCallType.PROPOSAL: ProposalDecision(
            personal_norm="ok", community_proposal=winning_text, operationalization="Publish a shared log."
        ),
        LLMCallType.NORM_COMPILER: NormCompilerOutput.model_validate(
            {"primitives": [{"type": "peer_observability"}]}
        ),
    }
    script[LLMCallType.VOTE] = lambda response_model, system, prompt: response_model(
        chosen_id="1"
    )
    fake_llm = FakeLLMClient(script)
    decisions = LLMDecisionSource(fake_llm, ModelConfig.default())
    compiler = NormCompiler(fake_llm)
    events = InMemoryEventSink()

    await run_norm_adoption(state, decisions, compiler, events)
    assert any(isinstance(p, PeerObservability) for p in state.active_norms)

    await decisions.decide_effort("a2", state)
    prompt = fake_llm.calls[-1][2]
    assert "a1:" in prompt and "a1 (you)" not in prompt
