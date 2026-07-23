"""Reflection: synthesize recent memories into a higher-level insight, written
back into the memory stream (build spec §3).
"""

from pydantic import BaseModel, Field

from genfishery.config.model_config import LLMCallType
from genfishery.llm.client import LLMClient
from genfishery.memory.bank import AgentMemoryBank

REFLECTION_SYSTEM_PROMPT = (
    "You are a villager reflecting on your recent experiences fishing from a shared lake."
)


class ReflectionInsight(BaseModel):
    insight: str
    importance: float = Field(ge=1.0, le=10.0)


async def reflect(llm: LLMClient, bank: AgentMemoryBank, *, current_round: int, recent_k: int = 20) -> None:
    recent = bank.retrieve_recent(recent_k)
    memory_lines = "\n".join(f"- (round {record.round}) {record.content}" for record in recent)
    prompt = (
        f"You are villager {bank.agent_id}. Here are your recent memories, oldest first:\n"
        f"{memory_lines}\n\n"
        "Synthesize these into one higher-level insight about the community, the "
        "shared lake, or your own strategy going forward. Also rate how important "
        "this insight is for you to remember (1-10)."
    )
    result = await llm.structured_call(
        call_type=LLMCallType.REFLECTION,
        system=REFLECTION_SYSTEM_PROMPT,
        prompt=prompt,
        response_model=ReflectionInsight,
    )
    bank.add_memory(result.insight, importance=result.importance, kind="reflection", round=current_round)
    bank.mark_reflected()
