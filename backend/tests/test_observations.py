from genfishery.models.events import Event, EventType
from genfishery.sim.observations import render_event_as_observation


def test_strategy_declared_renders_first_person():
    event = Event.create(
        fishery_id="f", round=1, phase="strategy", type=EventType.STRATEGY_DECLARED,
        actor_id="a1", payload={"effort": 0.42},
    )
    assert render_event_as_observation(event, "a1") == "I decided to fish with effort 0.42 this round."


def test_harvest_resolved_renders_only_viewers_own_catch():
    event = Event.create(
        fishery_id="f", round=1, phase="harvest", type=EventType.HARVEST_RESOLVED,
        payload={"harvests": {"a1": 3.5, "a2": 1.2}, "regrown_stock": 90.0, "pre_harvest_stock": 100.0, "post_harvest_stock": 95.0},
    )
    assert "3.50" in render_event_as_observation(event, "a1")
    assert "1.20" in render_event_as_observation(event, "a2")


def test_harvest_resolved_returns_none_for_agent_with_no_harvest_entry():
    event = Event.create(
        fishery_id="f", round=1, phase="harvest", type=EventType.HARVEST_RESOLVED,
        payload={"harvests": {"a1": 3.5}, "regrown_stock": 90.0},
    )
    assert render_event_as_observation(event, "a3") is None


def test_penalty_applied_renders_first_person_for_target_and_third_person_for_bystander():
    event = Event.create(
        fishery_id="f", round=1, phase="penalise", type=EventType.PENALTY_APPLIED,
        target_id="a2",
        payload={"penalty_type": "forfeit", "amount": 5.0, "destination": "pool", "trigger": "exceed_cap"},
    )
    assert render_event_as_observation(event, "a2") == "I was penalised 5.00 for exceeding the exceed cap rule."
    assert (
        render_event_as_observation(event, "a3")
        == "a2 was penalised 5.00 for exceeding the exceed cap rule."
    )


def test_agent_starved_returns_none_for_the_starved_agent_itself():
    event = Event.create(
        fishery_id="f", round=1, phase="starvation", type=EventType.AGENT_STARVED,
        target_id="a2", payload={"payoff": -1.0, "reason": "underharvest"},
    )
    assert render_event_as_observation(event, "a2") is None
    assert (
        render_event_as_observation(event, "a3")
        == "a2 starved and left the community, due to not catching enough fish."
    )


def test_agent_starved_by_punishment_renders_a_different_cause():
    event = Event.create(
        fishery_id="f", round=1, phase="starvation", type=EventType.AGENT_STARVED,
        target_id="a2", payload={"payoff": -1.0, "reason": "punished"},
    )
    assert (
        render_event_as_observation(event, "a3")
        == "a2 starved and left the community, due to a punishment penalty."
    )


def test_fishery_collapsed_states_resource_depletion_reason():
    event = Event.create(
        fishery_id="f", round=1, phase="collapse", type=EventType.FISHERY_COLLAPSED,
        payload={"stock": 0.0, "n_alive": 3, "reasons": ["resource_depletion"]},
    )
    assert (
        render_event_as_observation(event, "a1")
        == "The lake's fishery collapsed this round, because the fish stock was depleted."
    )


def test_fishery_collapsed_states_population_loss_reason():
    event = Event.create(
        fishery_id="f", round=1, phase="collapse", type=EventType.FISHERY_COLLAPSED,
        payload={"stock": 50.0, "n_alive": 0, "reasons": ["population_loss"]},
    )
    assert (
        render_event_as_observation(event, "a1")
        == "The lake's fishery collapsed this round, because too many villagers starved and left."
    )


def test_fishery_collapsed_states_both_reasons():
    event = Event.create(
        fishery_id="f", round=1, phase="collapse", type=EventType.FISHERY_COLLAPSED,
        payload={"stock": 0.0, "n_alive": 0, "reasons": ["resource_depletion", "population_loss"]},
    )
    text = render_event_as_observation(event, "a1")
    assert "the fish stock was depleted" in text
    assert "too many villagers starved and left" in text


def test_migration_renders_first_person_for_departure_and_arrival():
    departure = Event.create(
        fishery_id="fishery_a", round=5, phase="migration", type=EventType.MIGRATION,
        actor_id="a1", payload={"direction": "departure", "to_fishery": "fishery_b"},
    )
    arrival = Event.create(
        fishery_id="fishery_b", round=1, phase="migration", type=EventType.MIGRATION,
        actor_id="a1", payload={"direction": "arrival", "from_fishery": "fishery_a"},
    )
    assert "I left" in render_event_as_observation(departure, "a1")
    assert "I arrived" in render_event_as_observation(arrival, "a1")
    # Bystanders left behind see a departure; bystanders in the target
    # fishery are told a new fisherman arrived -- different framings of the
    # same MIGRATION event type, depending on `direction`.
    assert "a1 migrated to another fishery" in render_event_as_observation(departure, "a2")
    assert "A new fisherman, a1, has arrived" in render_event_as_observation(arrival, "a2")


def test_operationalization_result_with_no_winners_is_unrenderable():
    event = Event.create(
        fishery_id="f", round=1, phase="operationalization_vote", type=EventType.OPERATIONALIZATION_RESULT,
        payload={"results": {}},
    )
    assert render_event_as_observation(event, "a1") is None
