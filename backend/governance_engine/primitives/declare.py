from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

from governance_engine.interfaces import EnforcementResult, Primitive


class DeclareParameters(BaseModel):
    timing: Literal["pre_round", "post_round", "within_N_hours"]
    content: Literal["intended_quota", "actual_take", "violation_observed"]
    visibility: Literal["public", "anonymous", "ledger_only"]


class DeclarePrimitive(Primitive):
    name = "declare"

    def __init__(self, parameters: dict):
        self.parameters = DeclareParameters(**parameters).model_dump()

    def required_phases(self) -> list[str]:
        return ["declare_phase"]

    def enforce(self, agent_id: str, action: dict, world_state: dict) -> EnforcementResult:
        declarations: dict = world_state.get("declarations", {})

        if agent_id in declarations:
            return EnforcementResult(
                permitted=True,
                adjusted_amount=action.get("amount", 0.0),
            )

        violation = {
            "type": "missing_declaration",
            "agent_id": agent_id,
            "primitive": "declare",
        }
        event = {
            "type": "declare_violation",
            "agent_id": agent_id,
            "trigger": "fail_declare",
        }
        return EnforcementResult(
            permitted=False,
            adjusted_amount=0.0,
            violations=[violation],
            events=[event],
        )

    def is_satisfied(self, world_state: dict) -> bool:
        agents = world_state.get("agent_ids", [])
        declarations = world_state.get("declarations", {})
        return all(a in declarations for a in agents)
