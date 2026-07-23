import pytest

from genfishery.models.events import Event, EventType, Visibility


def test_create_derives_fixed_visibility_for_known_type():
    event = Event.create(
        fishery_id="fishery-a",
        round=1,
        phase="harvest",
        type=EventType.AGENT_STARVED,
        actor_id=None,
        target_id="agent_3",
        payload={"payoff": -1.2},
    )
    assert event.visibility == Visibility.PUBLIC


def test_strategy_declared_is_actor_only_by_fixed_rule():
    event = Event.create(
        fishery_id="fishery-a",
        round=1,
        phase="strategy",
        type=EventType.STRATEGY_DECLARED,
        actor_id="agent_1",
        payload={"effort": 0.4},
    )
    assert event.visibility == Visibility.ACTOR_ONLY


def test_direct_construction_rejects_visibility_mismatch():
    with pytest.raises(ValueError):
        Event(
            fishery_id="fishery-a",
            round=1,
            phase="harvest",
            type=EventType.AGENT_STARVED,
            visibility=Visibility.ACTOR_ONLY,  # wrong: fixed rule says PUBLIC
            payload={},
        )


def test_direct_construction_allows_correct_fixed_visibility():
    event = Event(
        fishery_id="fishery-a",
        round=2,
        phase="vote",
        type=EventType.VOTE_RESULT,
        visibility=Visibility.PUBLIC,
        payload={"norm_id": "norm-1", "passed": True},
    )
    assert event.id is None
    assert event.created_at is None
