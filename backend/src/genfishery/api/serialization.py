"""FisheryState -> JSON snapshot for `GET /fisheries/{id}/state` (build spec §9).

Active norms render from their source natural-language text
(`state.norm_source_text`), not raw primitive JSON, per the UI requirements --
the raw primitive fields are included too (needed for the debug/click-through
view), but `source_text` is what a normal panel shows.
"""

from typing import Any

from genfishery.sim.state import FisheryState


def serialize_agent(state: FisheryState, agent_id: str) -> dict[str, Any]:
    agent = state.agents[agent_id]
    return {
        "agent_id": agent.agent_id,
        "alive": agent.alive,
        "payoff": agent.payoff,
        "last_effort": agent.last_effort,
        "last_harvest": agent.last_harvest,
        "personal_norm": state.agent_norms.get(agent.agent_id),
        "persona_type": state.persona_types.get(agent.agent_id),
        "persona_description": state.persona_descriptions.get(agent.agent_id),
    }


def serialize_state(state: FisheryState) -> dict[str, Any]:
    return {
        "fishery_id": state.config.fishery_id,
        "round": state.round,
        "stock": state.stock,
        "carrying_capacity": state.config.k,
        "collapsed": state.collapsed,
        "group_norm_text": state.group_norm_text,
        "roles": state.roles,
        "agents": [serialize_agent(state, agent_id) for agent_id in state.agents],
        "active_norms": [
            {
                "id": primitive.id,
                "type": primitive.type,
                "source_text": state.norm_source_text.get(primitive.id),
                "detail": primitive.model_dump(),
            }
            for primitive in state.active_norms
        ],
    }
