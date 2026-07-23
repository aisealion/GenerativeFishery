from genfishery.config.model_config import LLMCallType
from genfishery.llm.fake_client import FakeLLMClient
from genfishery.memory.bank import AgentMemoryBank, MemoryConfig
from genfishery.memory.importance import ImportanceRating, rate_importance
from genfishery.memory.reflection import ReflectionInsight, reflect
from genfishery.memory.writer import write_observation
from tests.conftest import fake_embedder


async def test_rate_importance_returns_scripted_score():
    llm = FakeLLMClient({LLMCallType.IMPORTANCE_RATING: ImportanceRating(score=7.0)})
    score = await rate_importance(llm, "agent_1", "I was punished for overfishing.")
    assert score == 7.0
    assert llm.calls[0][0] == LLMCallType.IMPORTANCE_RATING


async def test_rate_importance_prompt_states_agent_identity():
    llm = FakeLLMClient({LLMCallType.IMPORTANCE_RATING: ImportanceRating(score=7.0)})
    await rate_importance(llm, "agent_1", "I was punished for overfishing.")
    prompt = llm.calls[-1][2]
    assert "You are villager agent_1" in prompt


async def test_reflect_writes_insight_back_to_memory_and_resets_counter():
    bank = AgentMemoryBank("agent_1", fake_embedder, config=MemoryConfig(reflection_importance_threshold=5.0))
    bank.add_memory("caught some fish", importance=6.0, kind="observation", round=1)
    assert bank.should_reflect() is True

    llm = FakeLLMClient(
        {LLMCallType.REFLECTION: ReflectionInsight(insight="I should fish less.", importance=8.0)}
    )
    await reflect(llm, bank, current_round=2)

    assert bank.should_reflect() is False
    recent = bank.retrieve_recent(2)
    assert recent[-1].kind == "reflection"
    assert recent[-1].content == "I should fish less."


async def test_write_observation_rates_importance_and_auto_triggers_reflection():
    bank = AgentMemoryBank("agent_1", fake_embedder, config=MemoryConfig(reflection_importance_threshold=5.0))
    llm = FakeLLMClient(
        {
            LLMCallType.IMPORTANCE_RATING: ImportanceRating(score=9.0),
            LLMCallType.REFLECTION: ReflectionInsight(insight="Punishment is working.", importance=6.0),
        }
    )

    await write_observation(llm, bank, "agent_2 punished agent_3 for overharvesting", round=1)

    kinds = [r.kind for r in bank.retrieve_recent(2)]
    assert kinds == ["observation", "reflection"]
    assert bank.should_reflect() is False


async def test_reflect_prompt_states_agent_identity_and_memory_round_numbers():
    bank = AgentMemoryBank("agent_1", fake_embedder, config=MemoryConfig(reflection_importance_threshold=5.0))
    bank.add_memory("caught some fish", importance=6.0, kind="observation", round=3)
    llm = FakeLLMClient(
        {LLMCallType.REFLECTION: ReflectionInsight(insight="I should fish less.", importance=8.0)}
    )

    await reflect(llm, bank, current_round=5)

    prompt = llm.calls[-1][2]
    assert "You are villager agent_1" in prompt
    assert "(round 3) caught some fish" in prompt


async def test_write_observation_does_not_reflect_below_threshold():
    bank = AgentMemoryBank("agent_1", fake_embedder, config=MemoryConfig(reflection_importance_threshold=50.0))
    llm = FakeLLMClient({LLMCallType.IMPORTANCE_RATING: ImportanceRating(score=2.0)})

    await write_observation(llm, bank, "a quiet day of fishing", round=1)

    kinds = [r.kind for r in bank.retrieve_recent(5)]
    assert kinds == ["observation"]
