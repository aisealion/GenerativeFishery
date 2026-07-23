from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

from governance_engine.interfaces import EnforcementResult, Primitive


class RedistributeParameters(BaseModel):
    source: Literal["violator", "communal_fund", "pool"]
    destination: Literal["pool", "all_agents_equal", "communal_fund", "all_agents_proportional"]
    trigger: Literal["violation", "end_of_round", "threshold"]
    amount: Literal["excess_units", "fixed", "proportional"]


class RedistributePrimitive(Primitive):
    name = "redistribute"

    def __init__(self, parameters: dict):
        self.parameters = RedistributeParameters(**parameters).model_dump()

    def required_phases(self) -> list[str]:
        return []

    def _trigger_met(self, agent_id: str, world_state: dict) -> bool:
        p = RedistributeParameters(**self.parameters)
        if p.trigger == "end_of_round":
            return True
        if p.trigger == "violation":
            pending = world_state.get("pending_violations", {}).get(agent_id, [])
            return bool(pending)
        if p.trigger == "threshold":
            stock = world_state.get("stock", float("inf"))
            threshold = world_state.get("sustainable_threshold", 0.0)
            return stock < threshold
        return False

    def _compute_amount(self, agent_id: str, world_state: dict) -> float:
        p = RedistributeParameters(**self.parameters)
        if p.amount == "excess_units":
            violations = world_state.get("pending_violations", {}).get(agent_id, [])
            return sum(v.get("excess", 0.0) for v in violations)
        if p.amount == "fixed":
            return world_state.get("redistribute_fixed_amount", 0.0)
        if p.amount == "proportional":
            communal = world_state.get("communal_fund", 0.0)
            n_agents = max(len(world_state.get("agent_ids", [])), 1)
            return communal / n_agents
        return 0.0

    def enforce(self, agent_id: str, action: dict, world_state: dict) -> EnforcementResult:
        p = RedistributeParameters(**self.parameters)
        events: list[dict] = []

        if not self._trigger_met(agent_id, world_state):
            return EnforcementResult(permitted=True, adjusted_amount=action.get("amount", 0.0))

        total = self._compute_amount(agent_id, world_state)
        agent_ids: list[str] = world_state.get("agent_ids", [])
        n = max(len(agent_ids), 1)

        if p.destination == "all_agents_equal":
            per_agent = total / n
            deltas = {a: per_agent for a in agent_ids}
        elif p.destination == "all_agents_proportional":
            weights = world_state.get("agent_weights", {})
            total_w = sum(weights.values()) or 1.0
            deltas = {a: total * weights.get(a, 0.0) / total_w for a in agent_ids}
        else:
            deltas = {}

        events.append({
            "type": "redistribution_applied",
            "source": p.source,
            "destination": p.destination,
            "total_amount": total,
            "per_agent_deltas": deltas,
            "agent_id": agent_id,
            "primitive": "redistribute",
        })

        return EnforcementResult(
            permitted=True,
            adjusted_amount=action.get("amount", 0.0),
            events=events,
        )

    def is_satisfied(self, world_state: dict) -> bool:
        return True
