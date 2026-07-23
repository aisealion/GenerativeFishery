from __future__ import annotations

import random
from typing import Literal

from pydantic import BaseModel

from governance_engine.interfaces import EnforcementResult, Primitive


class MonitorParameters(BaseModel):
    method: Literal["peer", "rotating_role", "central_board", "automated", "random_audit"]
    frequency: Literal["every_round", "random", "triggered"]
    target: Literal["individual", "total_pool"]
    ledger: bool = True


class MonitorPrimitive(Primitive):
    name = "monitor"

    def __init__(self, parameters: dict):
        self.parameters = MonitorParameters(**parameters).model_dump()

    def required_phases(self) -> list[str]:
        return ["monitor_phase"]

    def enforce(self, agent_id: str, action: dict, world_state: dict) -> EnforcementResult:
        p = MonitorParameters(**self.parameters)
        events: list[dict] = []

        if p.method == "random_audit" and random.random() < 0.3:
            events.append({
                "type": "audit_triggered",
                "agent_id": agent_id,
                "primitive": "monitor",
            })

        return EnforcementResult(
            permitted=True,
            adjusted_amount=action.get("amount", 0.0),
            events=events,
        )

    def is_satisfied(self, world_state: dict) -> bool:
        return bool(world_state.get("monitor_report_submitted", False))
