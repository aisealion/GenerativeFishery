from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel

from governance_engine.interfaces import EnforcementResult, Primitive


class CapParameters(BaseModel):
    basis: Literal["fixed_units", "pct_of_stock", "pct_of_total_catch", "sustainable_yield"]
    value: float
    scope: Literal["per_agent_per_round", "cumulative_over_N_rounds", "total_pool_per_round"]
    dynamic: bool = False
    N: Optional[int] = None


class CapPrimitive(Primitive):
    name = "cap"

    def __init__(self, parameters: dict):
        self.parameters = CapParameters(**parameters).model_dump()

    def required_phases(self) -> list[str]:
        return []

    def _compute_cap(self, world_state: dict) -> float:
        p = CapParameters(**self.parameters)
        stock = world_state.get("stock", 0.0)
        capacity = world_state.get("capacity", 1.0)
        total_catch = world_state.get("total_catch", 0.0)
        sustainable = world_state.get("sustainable_threshold", capacity * 0.4)

        if p.basis == "fixed_units":
            return p.value
        elif p.basis == "pct_of_stock":
            return stock * p.value
        elif p.basis == "pct_of_total_catch":
            return total_catch * p.value
        elif p.basis == "sustainable_yield":
            return sustainable * p.value
        return p.value

    def enforce(self, agent_id: str, action: dict, world_state: dict) -> EnforcementResult:
        p = CapParameters(**self.parameters)
        cap_value = self._compute_cap(world_state) if p.dynamic else self._compute_cap(world_state)
        amount = action.get("amount", 0.0)

        # Always publish cap info so the orchestrator can build rich history messages
        world_state["cap_info"] = {
            "cap_value": cap_value,
            "basis": p.basis,
            "param_value": p.value,
        }

        if amount <= cap_value:
            return EnforcementResult(permitted=True, adjusted_amount=amount)

        violation = {
            "type": "cap_exceeded",
            "agent_id": agent_id,
            "requested": amount,
            "cap": cap_value,
            "excess": amount - cap_value,
            "primitive": "cap",
        }
        event = {
            "type": "cap_violation",
            "agent_id": agent_id,
            "excess": amount - cap_value,
            "trigger": "exceed_cap",
        }
        return EnforcementResult(
            permitted=False,
            adjusted_amount=cap_value,
            violations=[violation],
            events=[event],
        )

    def is_satisfied(self, world_state: dict) -> bool:
        cap_value = self._compute_cap(world_state)
        return world_state.get("total_catch", 0.0) <= cap_value
