from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel

from governance_engine.interfaces import EnforcementResult, Primitive


class AdjustParameters(BaseModel):
    trigger: Literal["collective_threshold_breached", "end_of_round", "cumulative_over_N_rounds"]
    target: Literal["individual_quota", "pool_cap", "group_cap"]
    direction: Literal["reduce", "restore", "recalculate"]
    value: float
    restore_condition: Optional[str] = None


class AdjustPrimitive(Primitive):
    name = "adjust"

    def __init__(self, parameters: dict):
        self.parameters = AdjustParameters(**parameters).model_dump()

    def required_phases(self) -> list[str]:
        return []

    def _trigger_met(self, world_state: dict) -> bool:
        p = AdjustParameters(**self.parameters)
        if p.trigger == "end_of_round":
            return True
        if p.trigger == "collective_threshold_breached":
            stock = world_state.get("stock", float("inf"))
            threshold = world_state.get("sustainable_threshold", 0.0)
            return stock < threshold
        if p.trigger == "cumulative_over_N_rounds":
            return world_state.get("cumulative_threshold_breached", False)
        return False

    def enforce(self, agent_id: str, action: dict, world_state: dict) -> EnforcementResult:
        p = AdjustParameters(**self.parameters)
        events: list[dict] = []

        if self._trigger_met(world_state):
            events.append({
                "type": "quota_adjusted",
                "agent_id": agent_id,
                "target": p.target,
                "direction": p.direction,
                "value": p.value,
                "restore_condition": p.restore_condition,
                "primitive": "adjust",
            })

        return EnforcementResult(
            permitted=True,
            adjusted_amount=action.get("amount", 0.0),
            events=events,
        )

    def is_satisfied(self, world_state: dict) -> bool:
        return not self._trigger_met(world_state)
