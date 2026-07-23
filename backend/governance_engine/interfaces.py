from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


class Resource(ABC):
    @abstractmethod
    def get_state(self) -> dict:
        ...

    @abstractmethod
    def apply_action(self, agent_id: str, amount: float) -> dict:
        ...

    @abstractmethod
    def get_sustainable_threshold(self) -> float:
        ...

    @abstractmethod
    def is_collapsed(self) -> bool:
        ...


class GovernanceAgent(ABC):
    role_stack: list[str] = []

    @property
    @abstractmethod
    def id(self) -> str:
        ...

    @abstractmethod
    def get_base_context(self) -> dict:
        ...

    @abstractmethod
    def receive_prompt(self, prompt: str) -> str:
        ...

    @abstractmethod
    def get_memory(self, query: str) -> list[dict]:
        ...

    @abstractmethod
    def write_memory(self, event: dict) -> None:
        ...


@dataclass
class EnforcementResult:
    permitted: bool
    adjusted_amount: float
    violations: list[dict] = field(default_factory=list)
    events: list[dict] = field(default_factory=list)


class Primitive(ABC):
    parameters: dict = {}

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @abstractmethod
    def required_phases(self) -> list[str]:
        ...

    @abstractmethod
    def enforce(self, agent_id: str, action: dict, world_state: dict) -> EnforcementResult:
        ...

    @abstractmethod
    def is_satisfied(self, world_state: dict) -> bool:
        ...


class PhaseTemplate(ABC):
    prompt_template: str = ""
    target: str = "all"
    collect: str = "free_text"

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @abstractmethod
    def on_complete(self, responses: dict[str, str], world_state: dict) -> list[dict]:
        ...


class DomainAdapter(ABC):
    @property
    @abstractmethod
    def resource(self) -> Resource:
        ...

    @abstractmethod
    def get_agents(self) -> list[GovernanceAgent]:
        ...

    @abstractmethod
    def is_collapse_condition_met(self, world_state: dict) -> bool:
        ...

    @abstractmethod
    def get_prompt_vocabulary(self) -> dict[str, str]:
        ...

    @abstractmethod
    def get_primitive_whitelist(self) -> list[str]:
        ...

    @abstractmethod
    def on_cycle_complete(self, world_state: dict) -> None:
        ...

    def build_action_prompt(self, agent: "GovernanceAgent", world_state: dict) -> str:
        """Build the action-decision prompt for one agent. Override for rich domain-specific prompts."""
        vocab = self.get_prompt_vocabulary()
        return f"State your {vocab.get('action', 'act')} amount for this round as a single number."

    def get_governing_agents(self) -> list["GovernanceAgent"]:
        """Return only agents subject to governance (excludes outsiders)."""
        return [a for a in self.get_agents() if not getattr(a, "is_outsider", False)]
