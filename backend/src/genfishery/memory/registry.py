"""Cross-fishery memory ownership (build spec §3, §7).

An `AgentMemoryBank` belongs to the agent, not to any single fishery -- this
registry is the thing that's actually shared between fisheries (constructed
once for the whole process/simulation), so migration is just re-keying which
fishery is currently allowed to read/write a given agent's bank, never
copying or resetting it.
"""

from collections.abc import Callable

import numpy as np

from genfishery.memory.bank import AgentMemoryBank, MemoryConfig


class MemoryBankRegistry:
    def __init__(
        self, embedder: Callable[[str], np.ndarray], config: MemoryConfig | None = None
    ) -> None:
        self._embedder = embedder
        self._config = config
        self._banks: dict[str, AgentMemoryBank] = {}

    def get_or_create(self, agent_id: str) -> AgentMemoryBank:
        bank = self._banks.get(agent_id)
        if bank is None:
            bank = AgentMemoryBank(agent_id, self._embedder, config=self._config)
            self._banks[agent_id] = bank
        return bank

    def __contains__(self, agent_id: str) -> bool:
        return agent_id in self._banks
