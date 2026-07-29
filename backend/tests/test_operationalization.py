"""Tests for the standalone operationalization pipeline: a superseded,
disabled-by-default alternate design (agents optionally propose how to put a
just-won policy into practice, an LLM classifier clusters those suggestions
by aspect, agents vote per-cluster) that `run_norm_adoption` no longer
calls. Kept defined and covered here since the functions still exist; the
integration tests near the bottom instead confirm the design that replaced
it -- every `ProposalDecision` now bundles its own operationalization
suggestion directly (see `test_governance.py` for the full
propose/vote/compile integration).
"""

import pytest

from genfishery.config.fishery_config import FisheryConfig
from genfishery.config.model_config import LLMCallType, ModelConfig
from genfishery.llm.fake_client import FakeLLMClient
from genfishery.models.events import EventType, Visibility
from genfishery.sim.decisions import (
    OperationalizationCluster,
    OperationalizationProposalDecision,
    OperationalizationSuggestion,
    ProposalDecision,
    build_operationalization_proposal_prompt,
    build_operationalization_vote_prompt,
    build_operationalization_vote_response_model,
)
from genfishery.sim.engine import (
    build_augmented_policy_text,
    run_norm_adoption,
    run_operationalization_propose_phase,
    run_operationalization_vote_phase,
)
from genfishery.sim.events_sink import InMemoryEventSink
from genfishery.sim.norm_compiler import NormCompiler, NormCompilerOutput, OperationalizationClassifierOutput
from genfishery.sim.observations import render_event_as_observation
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


# --- run_operationalization_propose_phase -------------------------------------


class StubProposalSource:
    def __init__(self, decisions_by_agent: dict[str, OperationalizationProposalDecision]):
        self._decisions = decisions_by_agent

    async def decide_operationalization_proposal(self, agent_id, state, *, raw_text):
        return self._decisions[agent_id]


async def test_propose_phase_assigns_sequential_ids_and_skips_declines():
    state = FisheryState.initial(make_config())
    decisions = StubProposalSource(
        {
            "a1": OperationalizationProposalDecision(aspect="timing", suggestion="Fish only at dawn."),
            "a2": OperationalizationProposalDecision(),  # declines
            "a3": OperationalizationProposalDecision(aspect="penalty", suggestion="Deduct 2 payoff."),
        }
    )
    events = InMemoryEventSink()

    suggestions = await run_operationalization_propose_phase(
        state, decisions, events, raw_text="Fish sustainably."
    )

    assert [s.suggestion_id for s in suggestions] == ["1", "2"]
    assert [s.agent_id for s in suggestions] == ["a1", "a3"]
    assert suggestions[0].aspect_label == "timing"
    assert suggestions[1].suggestion_text == "Deduct 2 payoff."

    proposed_events = [e for e in events.events if e.type == EventType.OPERATIONALIZATION_PROPOSED]
    assert len(proposed_events) == 2  # a2's decline produced no event
    assert proposed_events[0].actor_id == "a1"
    assert proposed_events[0].visibility == Visibility.PUBLIC


async def test_propose_phase_treats_blank_suggestion_as_a_decline():
    state = FisheryState.initial(make_config())
    decisions = StubProposalSource(
        {
            "a1": OperationalizationProposalDecision(aspect="timing", suggestion=""),
            "a2": OperationalizationProposalDecision(aspect=None, suggestion="Something"),
            "a3": OperationalizationProposalDecision(),
        }
    )
    events = InMemoryEventSink()

    suggestions = await run_operationalization_propose_phase(state, decisions, events, raw_text="x")

    assert suggestions == []
    assert events.events == []


# --- NormCompiler.classify_operationalization_suggestions ----------------------


def make_suggestions() -> list[OperationalizationSuggestion]:
    return [
        OperationalizationSuggestion(
            suggestion_id="1", agent_id="a1", aspect_label="timing", suggestion_text="Fish only at dawn."
        ),
        OperationalizationSuggestion(
            suggestion_id="2", agent_id="a2", aspect_label="when", suggestion_text="Fish before sunrise."
        ),
        OperationalizationSuggestion(
            suggestion_id="3", agent_id="a3", aspect_label="penalty", suggestion_text="Deduct 2 payoff."
        ),
    ]


async def test_classify_returns_empty_without_any_llm_call_when_no_suggestions():
    compiler = NormCompiler(FakeLLMClient({}))  # no script -- would raise if actually called
    clusters = await compiler.classify_operationalization_suggestions([])
    assert clusters == []


async def test_classify_builds_clusters_from_llm_output():
    output = OperationalizationClassifierOutput.model_validate(
        {
            "clusters": [
                {"canonical_aspect": "timing", "suggestion_ids": ["1", "2"]},
                {"canonical_aspect": "penalty", "suggestion_ids": ["3"]},
            ]
        }
    )
    compiler = NormCompiler(FakeLLMClient({LLMCallType.OPERATIONALIZATION_CLASSIFIER: output}))

    clusters = await compiler.classify_operationalization_suggestions(make_suggestions())

    assert [c.cluster_id for c in clusters] == ["aspect_0", "aspect_1"]
    assert clusters[0].canonical_aspect == "timing"
    assert clusters[0].suggestion_ids == ["1", "2"]
    assert clusters[1].suggestion_ids == ["3"]


async def test_classify_drops_unknown_ids_and_empty_clusters():
    output = OperationalizationClassifierOutput.model_validate(
        {
            "clusters": [
                {"canonical_aspect": "timing", "suggestion_ids": ["1", "99"]},  # 99 doesn't exist
                {"canonical_aspect": "made up", "suggestion_ids": ["42"]},  # entirely invalid -> dropped
            ]
        }
    )
    compiler = NormCompiler(FakeLLMClient({LLMCallType.OPERATIONALIZATION_CLASSIFIER: output}))

    clusters = await compiler.classify_operationalization_suggestions(make_suggestions())

    assert len(clusters) == 1
    assert clusters[0].suggestion_ids == ["1"]


# --- run_operationalization_vote_phase ------------------------------------------


class StubVoteSource:
    def __init__(self, choices_by_agent: dict[str, dict[str, str]]):
        self._choices = choices_by_agent

    async def decide_operationalization_vote(self, agent_id, state, *, raw_text, clusters, suggestions):
        return self._choices[agent_id]


async def test_vote_phase_tallies_per_cluster_and_returns_winners():
    state = FisheryState.initial(make_config())
    suggestions = make_suggestions()[:2]  # "1" and "2", both in one cluster
    clusters = [OperationalizationCluster(cluster_id="aspect_0", canonical_aspect="timing", suggestion_ids=["1", "2"])]
    decisions = StubVoteSource(
        {"a1": {"aspect_0": "1"}, "a2": {"aspect_0": "1"}, "a3": {"aspect_0": "2"}}
    )
    events = InMemoryEventSink()

    results = await run_operationalization_vote_phase(
        state, decisions, events, raw_text="x", suggestions=suggestions, clusters=clusters
    )

    assert results["aspect_0"].suggestion_id == "1"  # 2 votes vs 1
    cast_events = [e for e in events.events if e.type == EventType.OPERATIONALIZATION_VOTE_CAST]
    assert len(cast_events) == 3
    assert all(e.visibility == Visibility.ACTOR_ONLY for e in cast_events)
    result_event = next(e for e in events.events if e.type == EventType.OPERATIONALIZATION_RESULT)
    assert result_event.visibility == Visibility.PUBLIC
    assert result_event.payload["results"]["aspect_0"]["winner"] == "Fish only at dawn."


async def test_vote_phase_all_abstain_yields_no_winner():
    state = FisheryState.initial(make_config())
    suggestions = make_suggestions()[:2]
    clusters = [OperationalizationCluster(cluster_id="aspect_0", canonical_aspect="timing", suggestion_ids=["1", "2"])]
    decisions = StubVoteSource(
        {"a1": {"aspect_0": "abstain"}, "a2": {"aspect_0": "abstain"}, "a3": {"aspect_0": "abstain"}}
    )
    events = InMemoryEventSink()

    results = await run_operationalization_vote_phase(
        state, decisions, events, raw_text="x", suggestions=suggestions, clusters=clusters
    )

    assert results["aspect_0"] is None
    result_event = next(e for e in events.events if e.type == EventType.OPERATIONALIZATION_RESULT)
    assert result_event.payload["results"]["aspect_0"]["winner"] is None


# --- build_augmented_policy_text ------------------------------------------------


def test_augmented_text_falls_back_to_raw_text_when_nothing_won():
    clusters = [OperationalizationCluster(cluster_id="aspect_0", canonical_aspect="timing", suggestion_ids=["1"])]
    text = build_augmented_policy_text("Fish sustainably.", clusters, {"aspect_0": None})
    assert text == "Fish sustainably."


def test_augmented_text_appends_winning_suggestions():
    clusters = [
        OperationalizationCluster(cluster_id="aspect_0", canonical_aspect="timing", suggestion_ids=["1"]),
        OperationalizationCluster(cluster_id="aspect_1", canonical_aspect="penalty", suggestion_ids=["2"]),
    ]
    winner_timing = OperationalizationSuggestion(
        suggestion_id="1", agent_id="a1", aspect_label="timing", suggestion_text="Fish only at dawn."
    )
    results = {"aspect_0": winner_timing, "aspect_1": None}

    text = build_augmented_policy_text("Fish sustainably.", clusters, results)

    assert text.startswith("Fish sustainably.\n\nOperationalization details agreed by the community:\n")
    assert "- timing: Fish only at dawn." in text
    # "penalty" had no winner (all abstained) -- it must not appear as a detail line.
    assert "- penalty:" not in text


# --- Full run_norm_adoption integration: the standalone pipeline is never called --


async def test_run_norm_adoption_never_invokes_the_standalone_pipeline():
    """The classifier/operationalization-vote call types are deliberately
    left unscripted -- this proves `run_norm_adoption` never calls
    `run_operationalization_propose_phase`/`classify_operationalization_
    suggestions`/`run_operationalization_vote_phase` anymore (it would raise
    `LLMStructuredCallError: no script provided` if it did). The bundled
    `ProposalDecision.operationalization` field is what reaches the compiler
    instead -- see `test_governance.py`'s norm-adoption tests for that.
    """
    state = FisheryState.initial(make_config())
    winning_text = "Fish sustainably."
    script = {
        LLMCallType.PROPOSAL: ProposalDecision(
            personal_norm="ok", community_proposal=winning_text, operationalization="Fish only at dawn."
        ),
        LLMCallType.VOTE: lambda response_model, system, prompt: response_model(chosen_id="1"),
        LLMCallType.NORM_COMPILER: NormCompilerOutput(primitives=[]),
    }
    fake_llm = FakeLLMClient(script)
    decisions = LLMDecisionSource(fake_llm, ModelConfig.default())
    compiler = NormCompiler(fake_llm)
    events = InMemoryEventSink()

    await run_norm_adoption(state, decisions, compiler, events)

    assert state.group_norm_text == winning_text
    compiler_call = next(c for c in fake_llm.calls if c[0] == LLMCallType.NORM_COMPILER)
    assert "Fish only at dawn." in compiler_call[2]
    assert not any(
        e.type
        in (
            EventType.OPERATIONALIZATION_PROPOSED,
            EventType.OPERATIONALIZATION_VOTE_CAST,
            EventType.OPERATIONALIZATION_RESULT,
        )
        for e in events.events
    )


# --- observations.py rendering --------------------------------------------------


def test_render_operationalization_proposed():
    from genfishery.models.events import Event

    event = Event.create(
        fishery_id="f", round=1, phase="operationalization_propose", type=EventType.OPERATIONALIZATION_PROPOSED,
        actor_id="a1", payload={"aspect": "timing", "suggestion": "Fish only at dawn."},
    )
    assert render_event_as_observation(event, "a1") == 'I suggested for "timing": "Fish only at dawn."'
    assert render_event_as_observation(event, "a2") == 'a1 suggested for "timing": "Fish only at dawn."'


def test_render_operationalization_vote_cast():
    from genfishery.models.events import Event

    event = Event.create(
        fishery_id="f", round=1, phase="operationalization_vote", type=EventType.OPERATIONALIZATION_VOTE_CAST,
        actor_id="a1", payload={"aspect": "timing", "choice": "Fish only at dawn."},
    )
    assert render_event_as_observation(event, "a1") == 'I voted "Fish only at dawn." for the aspect "timing".'


def test_render_operationalization_result():
    from genfishery.models.events import Event

    event = Event.create(
        fishery_id="f", round=1, phase="operationalization_vote", type=EventType.OPERATIONALIZATION_RESULT,
        payload={
            "results": {
                "aspect_0": {"aspect": "timing", "winner": "Fish only at dawn."},
                "aspect_1": {"aspect": "penalty", "winner": None},
            }
        },
    )
    text = render_event_as_observation(event, "a1")
    assert 'for "timing", the community chose: "Fish only at dawn."' in text
    assert 'for "penalty", no suggestion was chosen' in text


# --- decisions.py prompt builders -----------------------------------------------


def test_operationalization_proposal_prompt_states_raw_text_and_identity():
    state = FisheryState.initial(make_config())
    _, prompt = build_operationalization_proposal_prompt(
        state=state, viewer_id="a1", raw_text="Fish sustainably."
    )
    assert "You are villager a1" in prompt
    assert 'it is now the fishery\'s shared policy, and you are helping decide\nhow to actually put it into practice: "Fish sustainably."' in prompt
    assert "you don't have to propose anything this round" in prompt


def test_operationalization_vote_prompt_lists_clusters_and_choices():
    state = FisheryState.initial(make_config())
    suggestions = make_suggestions()[:2]
    clusters = [OperationalizationCluster(cluster_id="aspect_0", canonical_aspect="timing", suggestion_ids=["1", "2"])]
    _, prompt = build_operationalization_vote_prompt(
        state=state, viewer_id="a1", raw_text="Fish sustainably.", clusters=clusters, suggestions=suggestions
    )
    assert 'Aspect: "timing"' in prompt
    assert "[1] a1:" in prompt
    assert "[2] a2:" in prompt
    assert "abstain" in prompt


def test_operationalization_vote_response_model_constrains_choices_per_cluster():
    clusters = [
        OperationalizationCluster(cluster_id="aspect_0", canonical_aspect="timing", suggestion_ids=["1", "2"]),
        OperationalizationCluster(cluster_id="aspect_1", canonical_aspect="penalty", suggestion_ids=["3"]),
    ]
    model = build_operationalization_vote_response_model(clusters)

    valid = model(aspect_0="1", aspect_1="abstain")
    assert valid.aspect_0 == "1"

    with pytest.raises(Exception):
        model(aspect_0="not-a-valid-id", aspect_1="abstain")


# --- policies.py wiring -----------------------------------------------------------


async def test_llm_decision_source_operationalization_proposal_uses_correct_call_type():
    state = FisheryState.initial(make_config())
    fake_llm = FakeLLMClient(
        {LLMCallType.OPERATIONALIZATION_PROPOSAL: OperationalizationProposalDecision(aspect="a", suggestion="b")}
    )
    decisions = LLMDecisionSource(fake_llm, ModelConfig.default())

    decision = await decisions.decide_operationalization_proposal("a1", state, raw_text="Fish sustainably.")

    assert decision.aspect == "a"
    assert fake_llm.calls[-1][0] == LLMCallType.OPERATIONALIZATION_PROPOSAL
    assert "Fish sustainably." in fake_llm.calls[-1][2]


async def test_llm_decision_source_operationalization_vote_uses_correct_call_type():
    state = FisheryState.initial(make_config())
    clusters = [OperationalizationCluster(cluster_id="aspect_0", canonical_aspect="timing", suggestion_ids=["1"])]
    suggestions = make_suggestions()[:1]
    fake_llm = FakeLLMClient(
        {LLMCallType.OPERATIONALIZATION_VOTE: lambda response_model, system, prompt: response_model(aspect_0="1")}
    )
    decisions = LLMDecisionSource(fake_llm, ModelConfig.default())

    choices = await decisions.decide_operationalization_vote(
        "a1", state, raw_text="x", clusters=clusters, suggestions=suggestions
    )

    assert choices == {"aspect_0": "1"}
    assert fake_llm.calls[-1][0] == LLMCallType.OPERATIONALIZATION_VOTE
