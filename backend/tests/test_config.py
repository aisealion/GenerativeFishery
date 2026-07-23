from pathlib import Path

import yaml

from genfishery.config.fishery_config import FisheryConfig
from genfishery.config.model_config import HAIKU, SONNET, LLMCallType, ModelConfig, load_model_config

CONFIGS_DIR = Path(__file__).resolve().parents[1] / "configs"


def test_default_model_config_tiers_high_volume_calls_to_haiku():
    config = ModelConfig.default()
    assert config.for_call(LLMCallType.EFFORT_DECISION).model == HAIKU
    assert config.for_call(LLMCallType.NORM_COMPILER).model == SONNET


def test_load_model_config_from_repo_yaml_matches_defaults():
    config = load_model_config(CONFIGS_DIR / "models.yaml")
    assert config.for_call(LLMCallType.EFFORT_DECISION).model == HAIKU
    assert config.for_call(LLMCallType.REFLECTION).model == SONNET


def test_load_model_config_partial_override_falls_back_for_rest(tmp_path):
    override = tmp_path / "models.yaml"
    override.write_text(
        "calls:\n  effort_decision:\n    model: claude-sonnet-5\n"
    )
    config = load_model_config(override)
    assert config.for_call(LLMCallType.EFFORT_DECISION).model == "claude-sonnet-5"
    # Untouched call types keep their built-in defaults.
    assert config.for_call(LLMCallType.NORM_COMPILER).model == SONNET


def test_replication_fishery_config_sets_n_min_to_starting_population():
    config = FisheryConfig.model_validate(
        yaml.safe_load((CONFIGS_DIR / "fisheries" / "replication.yaml").read_text())
    )
    assert config.n_min == len(config.initial_agent_ids)
    assert config.r_min == 0.0
