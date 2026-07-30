"""Integration tests for the FastAPI layer (build spec step 8).

Uses the real Postgres container + real LISTEN/NOTIFY path (that's the whole
point of this step), but substitutes a scripted FakeLLMClient and the
lightweight test embedder so no Anthropic API key or sentence-transformers
download is needed, and rounds are fast/deterministic.
"""

import time

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from genfishery.api.app import create_app
from genfishery.config.fishery_config import FisheryConfig
from genfishery.config.model_config import LLMCallType
from genfishery.llm.fake_client import FakeLLMClient
from genfishery.memory.importance import ImportanceRating
from genfishery.sim.decisions import ProposalDecision
from genfishery.sim.norm_compiler import NormCompilerOutput
from tests.conftest import fake_embedder


def make_test_config() -> FisheryConfig:
    return FisheryConfig(
        fishery_id="test_live",
        alpha=0.1,
        r=0.5,
        k=50.0,
        initial_stock=50.0,
        consumption=1.0,
        initial_agent_ids=["a1", "a2", "a3"],
        r_min=0.0,
        n_min=1,
        max_rounds=None,
    )


def make_test_llm() -> FakeLLMClient:
    from genfishery.sim.decisions import EffortDecision

    return FakeLLMClient(
        {
            # alpha=0.1, stock=50 -> harvest = 0.1*0.3*50 = 1.5 per agent,
            # comfortably above consumption=1.0 so agents don't starve round 1.
            LLMCallType.EFFORT_DECISION: EffortDecision(effort=0.3),
            LLMCallType.IMPORTANCE_RATING: ImportanceRating(score=2.0),
            LLMCallType.PROPOSAL: ProposalDecision(
                personal_norm="stay the course",
                community_proposal="Keep fishing moderately.",
            ),
            LLMCallType.VOTE: lambda response_model, system, prompt: response_model(chosen_id="1"),
            LLMCallType.NORM_COMPILER: NormCompilerOutput(primitives=[]),
        }
    )


def make_test_app():
    return create_app(
        llm_client=make_test_llm(),
        fishery_configs=[make_test_config()],
        round_interval_seconds=0.05,
        embedder=fake_embedder,
        log_dir=None,  # don't write real log files during tests
    )


def test_get_state_returns_snapshot_matching_config():
    app = make_test_app()
    with TestClient(app) as client:
        response = client.get("/fisheries/test_live/state")
        assert response.status_code == 200
        body = response.json()
        assert body["fishery_id"] == "test_live"
        assert body["carrying_capacity"] == 50.0
        assert {a["agent_id"] for a in body["agents"]} == {"a1", "a2", "a3"}
        assert body["active_norms"] == []  # punishment is opt-in, none active by default


def test_get_state_unknown_fishery_id_returns_404():
    app = make_test_app()
    with TestClient(app) as client:
        response = client.get("/fisheries/does_not_exist/state")
        assert response.status_code == 404


def test_state_advances_over_time_as_background_runner_progresses():
    app = make_test_app()
    with TestClient(app) as client:
        first = client.get("/fisheries/test_live/state").json()
        # Each round makes many real Postgres inserts (one per event, one
        # connection each) -- a single round can take noticeably longer than
        # the 0.05s inter-round sleep, so allow generous wall-clock time.
        deadline = time.monotonic() + 10.0
        second = first
        while time.monotonic() < deadline and second["round"] <= first["round"]:
            time.sleep(0.2)
            second = client.get("/fisheries/test_live/state").json()
        assert second["round"] > first["round"]


def test_websocket_stream_receives_live_events():
    app = make_test_app()
    with TestClient(app) as client:
        with client.websocket_connect("/fisheries/test_live/stream") as websocket:
            message = websocket.receive_json(mode="text")
            assert message["fishery_id"] == "test_live"
            assert "type" in message
            assert "round" in message


def test_websocket_stream_unknown_fishery_id_closes_connection():
    app = make_test_app()
    with TestClient(app) as client:
        with pytest.raises(WebSocketDisconnect) as exc_info:
            with client.websocket_connect("/fisheries/does_not_exist/stream"):
                pass
        assert exc_info.value.code == 4404


def test_fishery_configs_env_var_selects_a_single_fishery(monkeypatch):
    from genfishery.api.app import _load_default_demo_configs

    monkeypatch.setenv("FISHERY_CONFIGS", "live_demo_a.yaml")
    configs = _load_default_demo_configs()
    assert [c.fishery_id for c in configs] == ["fishery_a"]


def test_fishery_configs_env_var_unset_defaults_to_single_fishery(monkeypatch):
    from genfishery.api.app import _load_default_demo_configs

    monkeypatch.delenv("FISHERY_CONFIGS", raising=False)
    configs = _load_default_demo_configs()
    assert [c.fishery_id for c in configs] == ["fishery_a"]


def test_fishery_configs_env_var_supports_comma_separated_list(monkeypatch):
    from genfishery.api.app import _load_default_demo_configs

    monkeypatch.setenv("FISHERY_CONFIGS", "live_demo_b.yaml, live_demo_a.yaml")
    configs = _load_default_demo_configs()
    assert [c.fishery_id for c in configs] == ["fishery_b", "fishery_a"]
