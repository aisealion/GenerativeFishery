from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

from governance_engine.interfaces import EnforcementResult, Primitive


class PenaliseParameters(BaseModel):
    trigger: Literal["exceed_cap", "fail_declare", "fail_monitor_duty"]
    type: Literal["forfeit", "fine_units", "quota_reduction", "temporary_ban", "proportional_fine"]
    value: float
    duration: int = 1
    destination: Literal["pool", "communal_fund", "redistribute_equal"]


class PenalisePrimitive(Primitive):
    name = "penalise"

    def __init__(self, parameters: dict):
        self.parameters = PenaliseParameters(**parameters).model_dump()

    def required_phases(self) -> list[str]:
        return []

    def enforce(self, agent_id: str, action: dict, world_state: dict) -> EnforcementResult:
        p = PenaliseParameters(**self.parameters)
        pending: list[dict] = world_state.get("pending_violations", {}).get(agent_id, [])

        matching = [v for v in pending if v.get("trigger") == p.trigger]
        if not matching:
            return EnforcementResult(
                permitted=True,
                adjusted_amount=action.get("amount", 0.0),
            )

        violation = matching[0]
        excess = violation.get("excess", action.get("amount", 0.0))

        if p.type == "proportional_fine":
            penalty_amount = excess * p.value
        elif p.type == "forfeit":
            penalty_amount = excess
        else:
            penalty_amount = p.value

        event = {
            "type": "penalty_applied",
            "agent_id": agent_id,
            "penalty_type": p.type,
            "amount": penalty_amount,
            "destination": p.destination,
            "duration": p.duration,
            "trigger": p.trigger,
            "primitive": "penalise",
        }
        return EnforcementResult(
            permitted=True,
            adjusted_amount=action.get("amount", 0.0),
            events=[event],
        )

    def is_satisfied(self, world_state: dict) -> bool:
        return not bool(world_state.get("pending_violations", {}))
