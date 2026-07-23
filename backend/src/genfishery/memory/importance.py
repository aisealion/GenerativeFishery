"""LLM-rated importance at write time (build spec §3)."""

from pydantic import BaseModel, Field

from genfishery.config.model_config import LLMCallType
from genfishery.llm.client import LLMClient

IMPORTANCE_SYSTEM_PROMPT = (
    "You rate how significant, surprising, or emotionally impactful a memory is "
    "for a villager living by a shared lake."
)


class ImportanceRating(BaseModel):
    score: float = Field(ge=1.0, le=10.0)


async def rate_importance(llm: LLMClient, agent_id: str, content: str) -> float:
    prompt = (
        f"You are villager {agent_id}. On a scale of 1 to 10, where 1 is purely mundane "
        "(e.g. a routine day of fishing at the usual effort) and 10 is extremely significant "
        "(e.g. a villager's death, a new community policy, being punished or punishing "
        "someone), rate the significance of this memory of yours:\n\n"
        f'"{content}"'
    )
    rating = await llm.structured_call(
        call_type=LLMCallType.IMPORTANCE_RATING,
        system=IMPORTANCE_SYSTEM_PROMPT,
        prompt=prompt,
        response_model=ImportanceRating,
    )
    return rating.score
