"""Compose importance-rating, memory write, and reflection triggering
(build spec §3) into the single call site the rest of the codebase uses.
"""

from genfishery.llm.client import LLMClient
from genfishery.memory.bank import AgentMemoryBank
from genfishery.memory.importance import rate_importance
from genfishery.memory.reflection import reflect


async def write_observation(llm: LLMClient, bank: AgentMemoryBank, content: str, *, round: int) -> None:
    importance = await rate_importance(llm, bank.agent_id, content)
    bank.add_memory(content, importance=importance, kind="observation", round=round)
    if bank.should_reflect():
        await reflect(llm, bank, current_round=round)
