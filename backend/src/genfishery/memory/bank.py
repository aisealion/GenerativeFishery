"""Per-agent memory stream (build spec §3).

Backed by Concordia's real `concordia.associative_memory.basic_associative_memory.
AssociativeMemoryBank` for storage, embedding, and base retrieval (project
decision: substrate-only -- Concordia's `AssociativeMemory`/`LastNObservations`
*components* require a full `EntityAgent`/Phase state machine we're not
adopting, since our own LangGraph round engine already owns orchestration and
our decisions are pydantic-structured, not free text). Recency + importance +
relevance weighted retrieval and reflection triggers are layered on top here,
since the Concordia memory bank itself only stores raw `text` + `embedding`.

An `AgentMemoryBank` is keyed by `agent_id`, not by fishery -- `get_state()`/
`restore()` are the whole mechanism behind migration (§3, §7): serialize out
of fishery A, `restore()` into a fresh instance handed to fishery B's roster.
No memory or reflection is lost or reset.
"""

from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from concordia.associative_memory.basic_associative_memory import AssociativeMemoryBank

from genfishery.memory.records import MemoryRecord, decode, encode


@dataclass
class MemoryConfig:
    # Smallville's original default cumulative-importance reflection trigger.
    reflection_importance_threshold: float = 150.0
    recency_decay_per_round: float = 0.995
    weight_recency: float = 1.0
    weight_importance: float = 1.0
    weight_relevance: float = 1.0


class AgentMemoryBank:
    def __init__(
        self,
        agent_id: str,
        embedder: Callable[[str], np.ndarray],
        config: MemoryConfig | None = None,
    ) -> None:
        self.agent_id = agent_id
        self._embedder = embedder
        self._bank = AssociativeMemoryBank(sentence_embedder=embedder, allow_duplicates=True)
        self.config = config or MemoryConfig()
        self._seq = 0
        self._importance_since_reflection = 0.0

    def add_memory(self, content: str, *, importance: float, kind: str, round: int) -> None:
        text = encode(seq=self._seq, round=round, importance=importance, kind=kind, content=content)
        self._bank.add(text)
        self._seq += 1
        if kind != "reflection":
            self._importance_since_reflection += importance

    def should_reflect(self) -> bool:
        return self._importance_since_reflection >= self.config.reflection_importance_threshold

    def mark_reflected(self) -> None:
        self._importance_since_reflection = 0.0

    def retrieve_recent(self, k: int) -> list[MemoryRecord]:
        return [decode(t) for t in self._bank.retrieve_recent(k)]

    def retrieve(self, query: str, *, k: int, current_round: int) -> list[MemoryRecord]:
        """Weighted recency + importance + relevance retrieval.

        Concordia's `retrieve_associative` only returns a top-k-by-cosine-alone
        ordering with no scores exposed, which isn't enough to combine with
        recency/importance -- so relevance is computed directly here via the
        same (L2-normalized) embedder against the bank's own dataframe, which
        is Concordia's own storage, just read through its public API.
        """
        df = self._bank.get_data_frame()
        if df.empty:
            return []

        records = [decode(t) for t in df["text"]]
        query_embedding = self._embedder(query)

        scored: list[tuple[float, MemoryRecord]] = []
        for record, embedding in zip(records, df["embedding"], strict=True):
            relevance = float(np.dot(query_embedding, embedding))
            age = max(current_round - record.round, 0)
            recency = self.config.recency_decay_per_round**age
            importance_norm = record.importance / 10.0
            score = (
                self.config.weight_recency * recency
                + self.config.weight_importance * importance_norm
                + self.config.weight_relevance * relevance
            )
            scored.append((score, record))

        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [record for _, record in scored[:k]]

    def get_state(self) -> dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "seq": self._seq,
            "importance_since_reflection": self._importance_since_reflection,
            "config": asdict(self.config),
            "memory_bank": self._bank.get_state(),
        }

    def set_state(self, state: dict[str, Any]) -> None:
        if state["agent_id"] != self.agent_id:
            raise ValueError(
                f"state belongs to agent {state['agent_id']!r}, not {self.agent_id!r}"
            )
        self._seq = state["seq"]
        self._importance_since_reflection = state["importance_since_reflection"]
        self.config = MemoryConfig(**state["config"])
        self._bank.set_state(state["memory_bank"])

    @classmethod
    def restore(cls, state: dict[str, Any], embedder: Callable[[str], np.ndarray]) -> "AgentMemoryBank":
        bank = cls(state["agent_id"], embedder)
        bank.set_state(state)
        return bank
