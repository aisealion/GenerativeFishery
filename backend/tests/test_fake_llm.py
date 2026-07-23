import pytest
from pydantic import BaseModel, Field

from genfishery.config.model_config import LLMCallType
from genfishery.llm.client import LLMStructuredCallError
from genfishery.llm.fake_client import FakeLLMClient


class EffortDecision(BaseModel):
    effort: float = Field(ge=0.0, le=1.0)


class VoteDecision(BaseModel):
    support: bool


async def test_fixed_script_returns_same_response_every_call():
    client = FakeLLMClient({LLMCallType.EFFORT_DECISION: EffortDecision(effort=0.3)})
    first = await client.structured_call(
        call_type=LLMCallType.EFFORT_DECISION,
        system="s",
        prompt="p",
        response_model=EffortDecision,
    )
    second = await client.structured_call(
        call_type=LLMCallType.EFFORT_DECISION,
        system="s",
        prompt="p",
        response_model=EffortDecision,
    )
    assert first.effort == second.effort == 0.3
    assert len(client.calls) == 2


async def test_queued_script_consumes_in_order():
    client = FakeLLMClient(
        {LLMCallType.EFFORT_DECISION: [EffortDecision(effort=0.1), EffortDecision(effort=0.9)]}
    )
    first = await client.structured_call(
        call_type=LLMCallType.EFFORT_DECISION, system="s", prompt="p1", response_model=EffortDecision
    )
    second = await client.structured_call(
        call_type=LLMCallType.EFFORT_DECISION, system="s", prompt="p2", response_model=EffortDecision
    )
    assert (first.effort, second.effort) == (0.1, 0.9)


async def test_queue_exhaustion_raises():
    client = FakeLLMClient({LLMCallType.VOTE: [VoteDecision(support=True)]})
    await client.structured_call(
        call_type=LLMCallType.VOTE, system="s", prompt="p", response_model=VoteDecision
    )
    with pytest.raises(LLMStructuredCallError):
        await client.structured_call(
            call_type=LLMCallType.VOTE, system="s", prompt="p", response_model=VoteDecision
        )


async def test_wrong_response_model_requested_raises():
    client = FakeLLMClient({LLMCallType.EFFORT_DECISION: EffortDecision(effort=0.5)})
    with pytest.raises(LLMStructuredCallError):
        await client.structured_call(
            call_type=LLMCallType.EFFORT_DECISION,
            system="s",
            prompt="p",
            response_model=VoteDecision,
        )


async def test_unscripted_call_type_raises():
    client = FakeLLMClient({})
    with pytest.raises(LLMStructuredCallError):
        await client.structured_call(
            call_type=LLMCallType.PROPOSAL, system="s", prompt="p", response_model=VoteDecision
        )
