"""Tests for Gupta et al. Table 2 persona initialization (build spec extension:
altruism_ratio-driven random assignment of an initial personal-strategy text).
"""

import random

import pytest
from pydantic import ValidationError

from genfishery.config.fishery_config import FisheryConfig
from genfishery.sim.personas import (
    ALTRUISTIC_TEMPLATES,
    SELFISH_TEMPLATES,
    assign_personas,
)
from genfishery.sim.state import FisheryState


def test_assign_personas_respects_exact_ratio_not_a_coin_flip():
    agent_ids = [f"a{i}" for i in range(10)]
    _, persona_types = assign_personas(agent_ids, 0.5, rng=random.Random(0))

    altruistic_count = sum(1 for t in persona_types.values() if t == "altruistic")
    selfish_count = sum(1 for t in persona_types.values() if t == "selfish")
    assert altruistic_count == 5
    assert selfish_count == 5


def test_assign_personas_rounds_fractional_split():
    agent_ids = [f"a{i}" for i in range(7)]
    _, persona_types = assign_personas(agent_ids, 0.5, rng=random.Random(0))
    altruistic_count = sum(1 for t in persona_types.values() if t == "altruistic")
    assert altruistic_count == round(7 * 0.5)  # 4


def test_assign_personas_ratio_zero_is_all_selfish():
    agent_ids = [f"a{i}" for i in range(5)]
    agent_norms, persona_types = assign_personas(agent_ids, 0.0, rng=random.Random(0))
    assert set(persona_types.values()) == {"selfish"}
    assert all(text in SELFISH_TEMPLATES for text in agent_norms.values())


def test_assign_personas_ratio_one_is_all_altruistic():
    agent_ids = [f"a{i}" for i in range(5)]
    agent_norms, persona_types = assign_personas(agent_ids, 1.0, rng=random.Random(0))
    assert set(persona_types.values()) == {"altruistic"}
    assert all(text in ALTRUISTIC_TEMPLATES for text in agent_norms.values())


def test_assign_personas_text_matches_declared_type():
    agent_ids = [f"a{i}" for i in range(10)]
    agent_norms, persona_types = assign_personas(agent_ids, 0.5, rng=random.Random(0))
    for agent_id, persona_type in persona_types.items():
        expected_pool = ALTRUISTIC_TEMPLATES if persona_type == "altruistic" else SELFISH_TEMPLATES
        assert agent_norms[agent_id] in expected_pool


def test_assign_personas_covers_every_agent_exactly_once():
    agent_ids = [f"a{i}" for i in range(10)]
    agent_norms, persona_types = assign_personas(agent_ids, 0.3, rng=random.Random(0))
    assert set(agent_norms) == set(agent_ids)
    assert set(persona_types) == set(agent_ids)


def test_assign_personas_deterministic_with_seeded_rng():
    agent_ids = [f"a{i}" for i in range(10)]
    norms_1, types_1 = assign_personas(agent_ids, 0.5, rng=random.Random(42))
    norms_2, types_2 = assign_personas(agent_ids, 0.5, rng=random.Random(42))
    assert norms_1 == norms_2
    assert types_1 == types_2


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
        n_min=1,
        max_rounds=None,
    )
    defaults.update(overrides)
    return FisheryConfig(**defaults)


def test_fishery_state_initial_assigns_personas_from_config_ratio():
    config = make_config(altruism_ratio=0.5)
    state = FisheryState.initial(config, seed=1)

    assert set(state.persona_types.values()) == {"altruistic", "selfish"}
    altruistic_count = sum(1 for t in state.persona_types.values() if t == "altruistic")
    assert altruistic_count == 2
    for agent_id, description in state.persona_descriptions.items():
        pool = ALTRUISTIC_TEMPLATES if state.persona_types[agent_id] == "altruistic" else SELFISH_TEMPLATES
        assert description in pool


def test_fishery_state_initial_leaves_agent_norms_unset():
    """persona_descriptions (fixed personality) and agent_norms (the
    agent's own evolving personal strategy, populated via ProposeNormPhase)
    are separate fields -- initializing personas must not seed agent_norms.
    """
    config = make_config(altruism_ratio=0.5)
    state = FisheryState.initial(config, seed=1)
    assert state.agent_norms == {}


def test_fishery_state_initial_default_altruism_ratio_is_half():
    config = make_config()
    assert config.altruism_ratio == 0.5


def test_fishery_state_initial_seed_is_reproducible():
    config = make_config(altruism_ratio=0.5)
    state_1 = FisheryState.initial(config, seed=7)
    state_2 = FisheryState.initial(config, seed=7)
    assert state_1.persona_descriptions == state_2.persona_descriptions
    assert state_1.persona_types == state_2.persona_types


def test_altruism_ratio_out_of_range_rejected():
    with pytest.raises(ValidationError):
        make_config(altruism_ratio=1.5)
