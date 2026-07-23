import pytest

from genfishery.memory.bank import AgentMemoryBank, MemoryConfig
from tests.conftest import fake_embedder


def make_bank(**config_overrides) -> AgentMemoryBank:
    config = MemoryConfig(**config_overrides) if config_overrides else None
    return AgentMemoryBank("agent_1", fake_embedder, config=config)


def test_retrieve_recent_returns_in_insertion_order():
    bank = make_bank()
    bank.add_memory("caught 3 fish", importance=2.0, kind="observation", round=1)
    bank.add_memory("voted for new policy", importance=4.0, kind="observation", round=2)
    bank.add_memory("was punished by agent_2", importance=8.0, kind="observation", round=3)

    recent = bank.retrieve_recent(2)
    assert [r.content for r in recent] == ["voted for new policy", "was punished by agent_2"]


def test_relevance_dominant_weighting_ranks_topical_match_first():
    bank = make_bank(weight_recency=0.0, weight_importance=0.0, weight_relevance=1.0)
    bank.add_memory("a storm passed over the lake", importance=1.0, kind="observation", round=1)
    bank.add_memory("I adjusted my fishing effort today", importance=1.0, kind="observation", round=10)

    top = bank.retrieve("How much effort should I put into fishing?", k=1, current_round=10)
    assert top[0].content == "I adjusted my fishing effort today"


def test_recency_dominant_weighting_ranks_newer_memory_first_regardless_of_topic():
    bank = make_bank(weight_recency=1.0, weight_importance=0.0, weight_relevance=0.0)
    bank.add_memory("fishing effort was high", importance=1.0, kind="observation", round=1)
    bank.add_memory("the weather was calm", importance=1.0, kind="observation", round=9)

    top = bank.retrieve("fishing effort", k=1, current_round=10)
    assert top[0].content == "the weather was calm"


def test_reflection_threshold_triggers_after_accumulated_importance():
    bank = make_bank(reflection_importance_threshold=10.0)
    assert bank.should_reflect() is False
    bank.add_memory("minor event", importance=6.0, kind="observation", round=1)
    assert bank.should_reflect() is False
    bank.add_memory("another minor event", importance=5.0, kind="observation", round=2)
    assert bank.should_reflect() is True

    bank.add_memory("synthesized insight", importance=9.0, kind="reflection", round=3)
    bank.mark_reflected()
    assert bank.should_reflect() is False
    # Reflection-kind writes must not themselves count toward the next trigger.
    bank.add_memory("another reflection", importance=20.0, kind="reflection", round=4)
    assert bank.should_reflect() is False


def test_state_round_trip_preserves_all_memory_and_bookkeeping():
    bank = make_bank(reflection_importance_threshold=10.0)
    bank.add_memory("caught fish", importance=3.0, kind="observation", round=1)
    bank.add_memory("insight about the lake", importance=9.0, kind="reflection", round=2)
    bank.mark_reflected()

    state = bank.get_state()
    restored = AgentMemoryBank.restore(state, fake_embedder)

    assert restored.agent_id == bank.agent_id
    assert restored.retrieve_recent(10) == bank.retrieve_recent(10)
    assert restored.should_reflect() == bank.should_reflect()
    # Confirm bookkeeping (seq, accumulated importance) also carried over by
    # writing one more memory to each and checking they stay in agreement.
    bank.add_memory("next", importance=1.0, kind="observation", round=3)
    restored.add_memory("next", importance=1.0, kind="observation", round=3)
    assert restored.retrieve_recent(1) == bank.retrieve_recent(1)


def test_set_state_rejects_state_for_a_different_agent():
    bank = make_bank()
    bank.add_memory("something", importance=1.0, kind="observation", round=1)
    state = bank.get_state()

    other_bank = AgentMemoryBank("agent_2", fake_embedder)
    with pytest.raises(ValueError):
        other_bank.set_state(state)
