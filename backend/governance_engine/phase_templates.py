from __future__ import annotations

import json
from collections import Counter

from governance_engine.db.models import AgentRoleRecord
from governance_engine.interfaces import PhaseTemplate
from governance_engine.utils import strip_json_fences


class _SafeFormatDict(dict):
    """Returns empty string for missing keys instead of raising KeyError."""
    def __missing__(self, key: str) -> str:
        return ""


def _render(template: str, vocab: dict, world_state: dict) -> str:
    ctx = _SafeFormatDict({**world_state, **vocab})
    return template.format_map(ctx)


class DeclarePhaseTemplate(PhaseTemplate):
    name = "declare_phase"
    target = "all"
    collect = "declaration"
    prompt_template = (
        "You are participating in a {resource} management system. "
        "Before this round begins, you must declare your {action} intention. "
        "State exactly how much you intend to {action} this round and your reasoning. "
        'Respond with a single JSON object: {{"intended_amount": float, "reasoning": str}}'
    )

    def __init__(self, vocabulary: dict, parameters: dict):
        self._vocab = vocabulary
        self._params = parameters

    def render_prompt(self, world_state: dict) -> str:
        return _render(self.prompt_template, self._vocab, world_state)

    def on_complete(self, responses: dict[str, str], world_state: dict) -> list[dict]:
        events: list[dict] = []
        declarations: dict = world_state.setdefault("declarations", {})

        for agent_id, raw in responses.items():
            try:
                data = json.loads(strip_json_fences(raw))
                intended = float(data.get("intended_amount", 0.0))
            except (json.JSONDecodeError, ValueError, TypeError):
                intended = 0.0
                data = {"raw_response": raw}

            declarations[agent_id] = {"intended_amount": intended, "raw": data}
            events.append({
                "type": "declaration_recorded",
                "agent_id": agent_id,
                "intended_amount": intended,
                "phase": self.name,
            })

        return events


class MonitorPhaseTemplate(PhaseTemplate):
    name = "monitor_phase"
    target = "role:monitor"
    collect = "report"
    prompt_template = (
        "You are the monitor this round. "
        "The following agents have declared their {action} amounts: {declarations_summary}. "
        "The {resource} cap this round is {cap_value}. "
        "Review each agent's declared amount and report any that exceed the cap. "
        'Respond with JSON: {{"violations": [{{"agent_id": str, "declared": float, "excess": float}}], "notes": str}}'
    )

    def __init__(self, vocabulary: dict, parameters: dict):
        self._vocab = vocabulary
        self._params = parameters

    def render_prompt(self, world_state: dict) -> str:
        decl = world_state.get("declarations", {})
        summary = ", ".join(
            f"{aid}: {v.get('intended_amount', '?')}" for aid, v in decl.items()
        )
        extra = {"declarations_summary": summary, "cap_value": world_state.get("cap_value", "unknown")}
        return _render(self.prompt_template, {**self._vocab, **extra}, world_state)

    def on_complete(self, responses: dict[str, str], world_state: dict) -> list[dict]:
        events: list[dict] = []
        pending: dict = world_state.setdefault("pending_violations", {})

        for _monitor_id, raw in responses.items():
            try:
                data = json.loads(strip_json_fences(raw))
                violations = data.get("violations", [])
            except (json.JSONDecodeError, TypeError):
                violations = []

            for v in violations:
                aid = v.get("agent_id", "")
                excess = v.get("excess", 0.0)
                pending.setdefault(aid, []).append({
                    "trigger": "exceed_cap",
                    "excess": excess,
                    "declared": v.get("declared", 0.0),
                    "source": "monitor_report",
                })
                events.append({
                    "type": "monitor_violation_recorded",
                    "agent_id": aid,
                    "excess": excess,
                    "phase": self.name,
                })

        world_state["monitor_report_submitted"] = True
        events.append({"type": "monitor_report_submitted", "phase": self.name})
        return events


class NominatePhaseTemplate(PhaseTemplate):
    name = "nominate_phase"
    target = "all"
    collect = "free_text"
    prompt_template = (
        "A {role} role has been established for this group. "
        "The {role} will be responsible for: {obligation_description}. "
        "Briefly explain why you would be effective in this role and whether you are willing to serve. "
        "If you decline, say so clearly."
    )

    def __init__(self, vocabulary: dict, parameters: dict):
        self._vocab = vocabulary
        self._params = parameters
        role = parameters.get("role", "monitor")
        obligation = parameters.get("obligation", "")
        self._vocab = {**vocabulary, "role": role, "obligation_description": obligation}

    def render_prompt(self, world_state: dict) -> str:
        return _render(self.prompt_template, self._vocab, world_state)

    def on_complete(self, responses: dict[str, str], world_state: dict) -> list[dict]:
        nominees: list[dict] = []
        for agent_id, text in responses.items():
            declined = any(word in text.lower() for word in ("decline", "no", "not willing", "pass"))
            if not declined:
                nominees.append({"agent_id": agent_id, "statement": text})

        world_state["nominees"] = nominees
        return [{"type": "nominations_collected", "nominees": nominees, "phase": self.name}]


class ElectPhaseTemplate(PhaseTemplate):
    name = "elect_phase"
    target = "all"
    collect = "vote"
    prompt_template = (
        "You must vote to elect a {role}. "
        "The nominees and their statements are: {nominee_summaries}. "
        "Vote for one nominee by responding with their agent ID only."
    )

    def __init__(self, vocabulary: dict, parameters: dict, session=None):
        self._vocab = vocabulary
        self._params = parameters
        self._session = session
        self._vocab = {**vocabulary, "role": parameters.get("role", "monitor")}

    def render_prompt(self, world_state: dict) -> str:
        nominees = world_state.get("nominees", [])
        summaries = " | ".join(
            f"{n['agent_id']}: {n['statement'][:80]}" for n in nominees
        )
        extra = {"nominee_summaries": summaries}
        return _render(self.prompt_template, {**self._vocab, **extra}, world_state)

    def on_complete(self, responses: dict[str, str], world_state: dict) -> list[dict]:
        tally: Counter = Counter()
        for _, vote in responses.items():
            tally[vote.strip()] += 1

        if not tally:
            return []

        winner, _ = tally.most_common(1)[0]
        role = self._params.get("role", "monitor")
        world_state.setdefault("role_assignments", {})[role] = winner

        current_round = world_state.get("round", 0)
        duration = self._params.get("duration", 1)

        if self._session is not None:
            record = AgentRoleRecord(
                agent_id=winner,
                role=role,
                assigned_round=current_round,
                expires_round=current_round + duration,
                source_rule_id=world_state.get("active_rule_id", 0),
            )
            self._session.add(record)
            self._session.commit()

        return [{
            "type": "role_assigned",
            "agent_id": winner,
            "role": role,
            "expires_round": current_round + duration,
            "phase": self.name,
        }]


class ExecuteRolePhaseTemplate(PhaseTemplate):
    collect = "report"
    prompt_template = (
        "You are the {role} this round. "
        "Your obligation is: {obligation_description}. "
        "Current {resource} state: {resource_state}. "
        "Agent actions this round: {actions_summary}. "
        'Submit your {role} report as JSON: {{"findings": list, "recommended_actions": list}}'
    )

    def __init__(self, vocabulary: dict, parameters: dict):
        self._vocab = vocabulary
        self._params = parameters
        role = parameters.get("role", "monitor")
        obligation = parameters.get("obligation", "")
        self._name = "execute_role_phase"
        self.target = f"role:{role}"
        self._vocab = {
            **vocabulary,
            "role": role,
            "obligation_description": obligation,
        }

    @property
    def name(self) -> str:
        return self._name

    def render_prompt(self, world_state: dict) -> str:
        resource_state = json.dumps(world_state.get("resource_state", {}))
        actions = world_state.get("agent_actions", {})
        actions_summary = ", ".join(f"{a}: {v}" for a, v in actions.items())
        extra = {"resource_state": resource_state, "actions_summary": actions_summary}
        return _render(self.prompt_template, {**self._vocab, **extra}, world_state)

    def on_complete(self, responses: dict[str, str], world_state: dict) -> list[dict]:
        events: list[dict] = []
        for agent_id, raw in responses.items():
            try:
                data = json.loads(strip_json_fences(raw))
            except (json.JSONDecodeError, TypeError):
                data = {"raw": raw}
            world_state.setdefault("role_reports", {})[agent_id] = data
            events.append({
                "type": "role_report_submitted",
                "agent_id": agent_id,
                "phase": self.name,
            })
        return events


class ReflectPhaseTemplate(PhaseTemplate):
    name = "reflect_phase"
    target = "all"
    collect = "free_text"
    prompt_template = (
        "The {resource} pool has collapsed or reached a critical threshold. "
        "Reviewing your history: {agent_history}. "
        "Propose one specific rule that would have prevented this outcome. "
        "Be concrete — specify limits, enforcement mechanism, and penalties. "
        "Your proposal:"
    )

    def __init__(self, vocabulary: dict, parameters: dict):
        self._vocab = vocabulary
        self._params = parameters

    def render_prompt(self, world_state: dict) -> str:
        history = world_state.get("agent_history", "No history available.")
        extra = {"agent_history": history}
        return _render(self.prompt_template, {**self._vocab, **extra}, world_state)

    def on_complete(self, responses: dict[str, str], world_state: dict) -> list[dict]:
        proposals = [
            {"agent_id": aid, "text": text} for aid, text in responses.items()
        ]
        world_state["proposals"] = proposals
        return [
            {
                "type": "proposal_submitted",
                "agent_id": p["agent_id"],
                "text": p["text"],
                "phase": self.name,
            }
            for p in proposals
        ]


class VotePhaseTemplate(PhaseTemplate):
    name = "vote_phase"
    target = "all"
    collect = "vote"
    prompt_template = (
        "The following rules have been proposed to prevent {resource} collapse: {proposals_summary}. "
        "Vote for the single rule you believe would be most effective. "
        "Respond with only the rule number (e.g. R1, R2)."
    )

    def __init__(self, vocabulary: dict, parameters: dict):
        self._vocab = vocabulary
        self._params = parameters

    def render_prompt(self, world_state: dict) -> str:
        proposals = world_state.get("proposals", [])
        lines = [f"R{i+1}: {p['text'][:120]}" for i, p in enumerate(proposals)]
        extra = {"proposals_summary": " || ".join(lines)}
        return _render(self.prompt_template, {**self._vocab, **extra}, world_state)

    def on_complete(self, responses: dict[str, str], world_state: dict) -> list[dict]:
        tally: Counter = Counter()
        n_voters = max(len(responses), 1)

        for _, vote in responses.items():
            tally[vote.strip()] += 1

        if not tally:
            return [{"type": "vote_complete", "winner": None, "phase": self.name}]

        winner_key, winner_count = tally.most_common(1)[0]
        winner = None
        if winner_count / n_voters > 0.5:
            proposals = world_state.get("proposals", [])
            try:
                idx = int(winner_key.lstrip("Rr")) - 1
                winner = proposals[idx]["text"] if 0 <= idx < len(proposals) else None
            except (ValueError, IndexError):
                winner = None

        world_state["winning_proposal"] = winner
        return [{
            "type": "vote_complete",
            "winner": winner,
            "tally": dict(tally),
            "phase": self.name,
        }]


PHASE_TEMPLATE_REGISTRY: dict[str, type] = {
    "declare_phase": DeclarePhaseTemplate,
    "monitor_phase": MonitorPhaseTemplate,
    "nominate_phase": NominatePhaseTemplate,
    "elect_phase": ElectPhaseTemplate,
    "execute_role_phase": ExecuteRolePhaseTemplate,
    "reflect_phase": ReflectPhaseTemplate,
    "vote_phase": VotePhaseTemplate,
}
