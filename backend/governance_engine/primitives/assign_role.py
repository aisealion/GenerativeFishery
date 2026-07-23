from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

from governance_engine.interfaces import EnforcementResult, Primitive


class AssignRoleParameters(BaseModel):
    role: Literal["monitor", "auditor", "ledger_keeper", "whistleblower"]
    selection: Literal["rotating", "random", "elected"]
    duration: int
    obligation: str


class AssignRolePrimitive(Primitive):
    name = "assign_role"

    def __init__(self, parameters: dict):
        self.parameters = AssignRoleParameters(**parameters).model_dump()

    def required_phases(self) -> list[str]:
        p = AssignRoleParameters(**self.parameters)
        if p.selection == "elected":
            return ["nominate_phase", "elect_phase", "execute_role_phase"]
        return ["execute_role_phase"]

    def enforce(self, agent_id: str, action: dict, world_state: dict) -> EnforcementResult:
        p = AssignRoleParameters(**self.parameters)
        role_assignments: dict = world_state.get("role_assignments", {})
        role_reports: dict = world_state.get("role_reports", {})
        events: list[dict] = []

        assigned_agent = role_assignments.get(p.role)
        if assigned_agent == agent_id:
            if agent_id not in role_reports:
                events.append({
                    "type": "role_obligation_unmet",
                    "agent_id": agent_id,
                    "role": p.role,
                    "obligation": p.obligation,
                    "trigger": "fail_monitor_duty",
                    "primitive": "assign_role",
                })

        return EnforcementResult(
            permitted=True,
            adjusted_amount=action.get("amount", 0.0),
            events=events,
        )

    def is_satisfied(self, world_state: dict) -> bool:
        p = AssignRoleParameters(**self.parameters)
        return p.role in world_state.get("role_assignments", {})
