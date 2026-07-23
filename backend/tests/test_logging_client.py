"""Tests for LoggingLLMClient (per-fishery prompt/response log files)."""

import pytest

from genfishery.config.model_config import LLMCallType
from genfishery.llm.client import LLMStructuredCallError
from genfishery.llm.fake_client import FakeLLMClient
from genfishery.llm.logging_client import LoggingLLMClient
from genfishery.sim.decisions import EffortDecision


async def test_successful_call_is_logged_and_still_returns_the_result(tmp_path):
    inner = FakeLLMClient({LLMCallType.EFFORT_DECISION: EffortDecision(effort=0.42)})
    log_path = tmp_path / "fishery_a.log"
    current_round = {"value": 3}
    client = LoggingLLMClient(
        inner, log_path, fishery_id="fishery_a", get_round=lambda: current_round["value"]
    )

    result = await client.structured_call(
        call_type=LLMCallType.EFFORT_DECISION,
        system="You are a villager.",
        prompt="Decide your effort.",
        response_model=EffortDecision,
    )

    assert result.effort == 0.42
    text = log_path.read_text()
    assert "fishery=fishery_a round=3 call_type=effort_decision" in text
    assert "You are a villager." in text
    assert "Decide your effort." in text
    assert '"effort": 0.42' in text


async def test_failed_call_is_logged_with_the_error_and_still_raises(tmp_path):
    inner = FakeLLMClient({})  # no script -> raises LLMStructuredCallError
    log_path = tmp_path / "fishery_b.log"
    client = LoggingLLMClient(inner, log_path, fishery_id="fishery_b", get_round=lambda: 9)

    with pytest.raises(LLMStructuredCallError):
        await client.structured_call(
            call_type=LLMCallType.IMPORTANCE_RATING,
            system="sys",
            prompt="rate this",
            response_model=EffortDecision,
        )

    text = log_path.read_text()
    assert "fishery=fishery_b round=9 call_type=importance_rating" in text
    assert "--- ERROR ---" in text
    assert "LLMStructuredCallError" in text


async def test_multiple_calls_append_in_order(tmp_path):
    inner = FakeLLMClient(
        {LLMCallType.EFFORT_DECISION: [EffortDecision(effort=0.1), EffortDecision(effort=0.9)]}
    )
    log_path = tmp_path / "fishery_a.log"
    client = LoggingLLMClient(inner, log_path, fishery_id="fishery_a", get_round=lambda: 1)

    await client.structured_call(
        call_type=LLMCallType.EFFORT_DECISION, system="s", prompt="first", response_model=EffortDecision
    )
    await client.structured_call(
        call_type=LLMCallType.EFFORT_DECISION, system="s", prompt="second", response_model=EffortDecision
    )

    text = log_path.read_text()
    assert text.index("first") < text.index("second")


async def test_creates_parent_directory_if_missing(tmp_path):
    inner = FakeLLMClient({LLMCallType.EFFORT_DECISION: EffortDecision(effort=0.5)})
    log_path = tmp_path / "nested" / "logs" / "fishery_a.log"
    client = LoggingLLMClient(inner, log_path, fishery_id="fishery_a", get_round=lambda: 0)

    await client.structured_call(
        call_type=LLMCallType.EFFORT_DECISION, system="s", prompt="p", response_model=EffortDecision
    )

    assert log_path.exists()
