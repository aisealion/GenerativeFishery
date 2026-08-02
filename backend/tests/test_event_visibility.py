from genfishery.models.events import Event, EventType, Visibility, visible_events_for


def make_event(**overrides) -> Event:
    defaults = dict(
        fishery_id="f",
        round=1,
        phase="p",
        type=EventType.HARVEST_RESOLVED,
        visibility=Visibility.PUBLIC,
        payload={},
    )
    defaults.update(overrides)
    return Event(**defaults)


def test_public_event_visible_to_anyone():
    event = make_event(visibility=Visibility.PUBLIC)
    assert event.is_visible_to("a1")
    assert event.is_visible_to("stranger")


def test_actor_only_visible_only_to_actor():
    event = make_event(
        type=EventType.STRATEGY_DECLARED, visibility=Visibility.ACTOR_ONLY, actor_id="a1"
    )
    assert event.is_visible_to("a1")
    assert not event.is_visible_to("a2")


def test_target_only_visible_only_to_target():
    # No EventType is currently fixed to TARGET_ONLY -- construct via
    # model_construct to unit-test `is_visible_to`'s handling of this
    # visibility level directly, independent of any specific event type.
    event = Event.model_construct(
        fishery_id="f",
        round=1,
        phase="p",
        type=EventType.MONITOR_REVIEW,
        visibility=Visibility.TARGET_ONLY,
        target_id="a2",
        payload={},
    )
    assert event.is_visible_to("a2")
    assert not event.is_visible_to("a1")


def test_actor_and_target_visible_to_both_only():
    # No EventType is currently fixed to ACTOR_AND_TARGET -- construct via
    # model_construct to unit-test `is_visible_to`'s handling of this
    # visibility level directly, independent of any specific event type.
    event = Event.model_construct(
        fishery_id="f",
        round=1,
        phase="p",
        type=EventType.MONITOR_REVIEW,
        visibility=Visibility.ACTOR_AND_TARGET,
        actor_id="a1",
        target_id="a2",
        payload={},
    )
    assert event.is_visible_to("a1")
    assert event.is_visible_to("a2")
    assert not event.is_visible_to("a3")


def test_visible_events_for_filters_and_preserves_order():
    events = [
        make_event(type=EventType.HARVEST_RESOLVED, visibility=Visibility.PUBLIC),
        make_event(type=EventType.STRATEGY_DECLARED, visibility=Visibility.ACTOR_ONLY, actor_id="a1"),
        make_event(type=EventType.STRATEGY_DECLARED, visibility=Visibility.ACTOR_ONLY, actor_id="a2"),
        make_event(type=EventType.VOTE_RESULT, visibility=Visibility.PUBLIC),
    ]
    visible = visible_events_for("a1", events)
    assert visible == [events[0], events[1], events[3]]
