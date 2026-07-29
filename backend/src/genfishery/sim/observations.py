"""Renders a visible `Event` into the natural-language observation text that
gets written to an agent's memory stream (build spec §3: "every observation
... stored as a timestamped natural-language record"). Deterministic string
formatting only -- no LLM involved in deciding *what* an event says, only in
rating its importance once it becomes a memory (see `memory.importance`).
"""

from genfishery.models.events import Event, EventType

_COLLAPSE_REASON_TEXT = {
    "resource_depletion": "the fish stock was depleted",
    "population_loss": "too many villagers starved and left",
    "underharvest_death": "a villager underharvested and starved",
}


def render_event_as_observation(event: Event, viewer_id: str) -> str | None:
    payload = event.payload

    if event.type == EventType.STRATEGY_DECLARED:
        return f"I decided to fish with effort {payload['effort']:.2f} this round."

    if event.type == EventType.DISCLOSURE_MADE:
        who = "I" if event.actor_id == viewer_id else event.actor_id
        return f"{who} disclosed {payload['field']}: {payload['value']}."

    if event.type == EventType.HARVEST_RESOLVED:
        my_harvest = payload["harvests"].get(viewer_id)
        if my_harvest is None:
            return None
        return (
            f"This round I caught {my_harvest:.2f} fish; the lake's stock is now "
            f"{payload['regrown_stock']:.2f}."
        )

    if event.type == EventType.CAP_EXCEEDED:
        who = "I" if event.actor_id == viewer_id else (event.actor_id or "The community")
        return (
            f"{who} exceeded the agreed catch limit this round "
            f"(caught {payload['observed']:.2f}, limit {payload['cap_value']:.2f})."
        )

    if event.type == EventType.PENALTY_APPLIED:
        cause = f"exceeding the {payload['trigger'].replace('_', ' ')} rule"
        if event.target_id == viewer_id:
            return f"I was penalised {payload['amount']:.2f} for {cause}."
        return f"{event.target_id} was penalised {payload['amount']:.2f} for {cause}."

    if event.type == EventType.QUOTA_ADJUSTED:
        return (
            f"The community's {payload['adjust_target'].replace('_', ' ')} was "
            f"{payload['direction']}d by {payload['value']:.2f}."
        )

    if event.type == EventType.REDISTRIBUTION_APPLIED:
        my_delta = payload.get("per_agent_deltas", {}).get(viewer_id)
        if my_delta:
            return f"I received {my_delta:.2f} from a redistribution ({payload['total_amount']:.2f} total)."
        if payload.get("per_agent_deltas"):
            return f"The community redistributed {payload['total_amount']:.2f} among villagers."
        return f"{payload['total_amount']:.2f} was redistributed to the {payload['destination']}."

    if event.type == EventType.MONITOR_REVIEW:
        who = "I was" if event.actor_id == viewer_id else f"{event.actor_id} was"
        return f"{who} audited this round."

    if event.type == EventType.AGENT_STARVED:
        if event.target_id == viewer_id:
            return None  # they're gone; nothing left to remember this into
        reason = payload.get("reason", "starvation")
        cause = "not catching enough fish" if reason == "underharvest" else "a punishment penalty"
        return f"{event.target_id} starved and left the community, due to {cause}."

    if event.type == EventType.PERSONAL_NORM_UPDATED:
        return f"I updated my personal strategy to: \"{payload['personal_norm']}\""

    if event.type == EventType.PROPOSAL_MADE:
        who = "I" if event.actor_id == viewer_id else event.actor_id
        return (
            f'{who} proposed: "{payload["proposal"]}" '
            f'(to operationalize: "{payload["operationalization"]}")'
        )

    if event.type == EventType.VOTE_CAST:
        return (
            f'I voted for: "{payload["choice"]}" '
            f'(to operationalize: "{payload["operationalization"]}")'
        )

    if event.type == EventType.VOTE_RESULT:
        return (
            f'The community voted; "{payload["winner"]}" won '
            f'(to operationalize: "{payload["operationalization"]}").'
        )

    if event.type == EventType.NORM_ADOPTED:
        return f'The community adopted a new policy: "{payload["raw_text"]}"'

    if event.type == EventType.NORM_COULD_NOT_COMPILE:
        return f'The community tried to adopt "{payload["raw_text"]}" but it could not be formalized.'

    if event.type == EventType.ROLE_ELECTION_CALLED:
        return f"An election was called for the role of {payload['role_name']}."

    if event.type == EventType.ROLE_ELECTED:
        who = "I" if payload.get("agent_id") == viewer_id else payload.get("agent_id")
        return f"{who} became {payload['role_name']} (via {payload['selection']})."

    if event.type == EventType.FISHERY_COLLAPSED:
        causes = [_COLLAPSE_REASON_TEXT[r] for r in payload.get("reasons", []) if r in _COLLAPSE_REASON_TEXT]
        if not causes:
            return "The lake's fishery collapsed this round."
        return f"The lake's fishery collapsed this round, because {' and '.join(causes)}."

    if event.type == EventType.MIGRATION:
        direction = payload.get("direction")
        if event.actor_id == viewer_id:
            if direction == "departure":
                return f"I left for another fishery ({payload.get('to_fishery')}) after this one collapsed."
            return f"I arrived here from another fishery ({payload.get('from_fishery')}), carrying my prior experience."
        if direction == "arrival":
            return f"A new fisherman, {event.actor_id}, has arrived in the fishery."
        return f"{event.actor_id} migrated to another fishery."

    return None
