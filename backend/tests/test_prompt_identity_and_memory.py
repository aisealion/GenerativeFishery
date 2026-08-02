"""Every prompt states the viewer's own identity, and (when a memory
registry is supplied) retrieves that agent's relevant memories back into the
prompt -- the retrieval half of the Smallville pattern that step 7 left
unwired.
"""

from genfishery.config.fishery_config import FisheryConfig
from genfishery.config.model_config import LLMCallType, ModelConfig
from genfishery.llm.fake_client import FakeLLMClient
from genfishery.memory.registry import MemoryBankRegistry
from genfishery.sim.decisions import EffortDecision
from genfishery.sim.policies import LLMDecisionSource
from genfishery.sim.state import FisheryState
from tests.conftest import fake_embedder


def make_config() -> FisheryConfig:
    return FisheryConfig(
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


def make_llm() -> FakeLLMClient:
    return FakeLLMClient({LLMCallType.EFFORT_DECISION: EffortDecision(effort=0.3)})


async def test_prompt_states_viewer_identity_and_defaults_to_own_observations_only():
    state = FisheryState.initial(make_config())
    for agent in state.agents.values():
        agent.last_effort = 0.2
    llm = make_llm()
    decisions = LLMDecisionSource(llm, ModelConfig.default())

    await decisions.decide_effort("a2", state)

    prompt = llm.calls[-1][2]
    assert "You are villager a2" in prompt
    assert "a2 (you)" in prompt
    # The roster still names a1 (identity/alive-status is always visible),
    # but their effort/payoff numbers are never disclosed to anyone else.
    assert "a1: active" in prompt
    assert "a1: effort=" not in prompt and "a1 (you)" not in prompt


async def test_roster_block_always_names_every_agent_and_marks_removed_ones():
    state = FisheryState.initial(make_config())
    state.agents["a3"].alive = False
    state.starvation_reasons["a3"] = "underharvest"
    llm = make_llm()
    decisions = LLMDecisionSource(llm, ModelConfig.default())

    await decisions.decide_effort("a2", state)

    prompt = llm.calls[-1][2]
    assert "Villagers in this fishery:" in prompt
    assert "- a1: active" in prompt
    assert "- a2 (you): active" in prompt
    assert "- a3: removed (underharvest)" in prompt


async def test_roster_block_tags_recent_migrant_then_reverts_to_active():
    state = FisheryState.initial(make_config())
    state.round = 5
    state.migration_arrival_round["a3"] = 4  # arrived last round -- within window
    llm = make_llm()
    decisions = LLMDecisionSource(llm, ModelConfig.default())

    await decisions.decide_effort("a2", state)
    prompt = llm.calls[-1][2]
    assert "- a3: arrived recently to the fishery" in prompt

    state.round = 20  # long past the newcomer window now
    await decisions.decide_effort("a2", state)
    prompt = llm.calls[-1][2]
    assert "- a3: active" in prompt
    assert "arrived recently" not in prompt


async def test_prompt_states_fixed_personality_separately_from_personal_strategy():
    state = FisheryState.initial(make_config(), seed=1)
    llm = make_llm()
    decisions = LLMDecisionSource(llm, ModelConfig.default())

    await decisions.decide_effort("a1", state)

    prompt = llm.calls[-1][2]
    description = state.persona_descriptions["a1"]
    assert f'Your personality: "{description}"' in prompt
    # Personal strategy has not been proposed yet -- it must not be seeded
    # with the persona template text (that would conflate a fixed trait with
    # the agent's own evolving belief).
    assert 'Your personal strategy: "(no norm has been established yet)"' in prompt
    assert f'Your personal strategy: "{description}"' not in prompt


async def test_personality_line_survives_after_agent_revises_personal_strategy():
    state = FisheryState.initial(make_config(), seed=1)
    description = state.persona_descriptions["a1"]
    state.agent_norms["a1"] = "I've changed my mind, let's fish carefully now."
    llm = make_llm()
    decisions = LLMDecisionSource(llm, ModelConfig.default())

    await decisions.decide_effort("a1", state)

    prompt = llm.calls[-1][2]
    assert f'Your personality: "{description}"' in prompt
    assert 'Your personal strategy: "I\'ve changed my mind, let\'s fish carefully now."' in prompt


async def test_no_memory_registry_omits_memories_section():
    state = FisheryState.initial(make_config())
    llm = make_llm()
    decisions = LLMDecisionSource(llm, ModelConfig.default())  # no memory_registry

    await decisions.decide_effort("a1", state)

    prompt = llm.calls[-1][2]
    assert "Relevant memories" not in prompt


async def test_empty_memory_bank_omits_memories_section():
    state = FisheryState.initial(make_config())
    llm = make_llm()
    registry = MemoryBankRegistry(fake_embedder)
    decisions = LLMDecisionSource(llm, ModelConfig.default(), registry)

    await decisions.decide_effort("a1", state)

    prompt = llm.calls[-1][2]
    assert "Relevant memories" not in prompt


async def test_populated_memory_bank_is_retrieved_into_the_prompt():
    state = FisheryState.initial(make_config())
    state.round = 5
    llm = make_llm()
    registry = MemoryBankRegistry(fake_embedder)
    bank = registry.get_or_create("a1")
    bank.add_memory(
        "I was punished by a2 for overfishing, losing 2.00.",
        importance=8.0,
        kind="observation",
        round=3,
    )
    decisions = LLMDecisionSource(llm, ModelConfig.default(), registry)

    await decisions.decide_effort("a1", state)

    prompt = llm.calls[-1][2]
    assert "Relevant memories from your past experience:" in prompt
    assert "I was punished by a2 for overfishing" in prompt
    assert "(round 3)" in prompt


async def test_memory_retrieval_is_per_agent_not_shared():
    state = FisheryState.initial(make_config())
    llm = make_llm()
    registry = MemoryBankRegistry(fake_embedder)
    registry.get_or_create("a1").add_memory(
        "a1's private memory about fishing effort.", importance=9.0, kind="observation", round=1
    )
    decisions = LLMDecisionSource(llm, ModelConfig.default(), registry)

    await decisions.decide_effort("a2", state)

    prompt = llm.calls[-1][2]
    assert "a1's private memory" not in prompt
