from __future__ import annotations

import json
from datetime import datetime

from loguru import logger
from sqlmodel import Session, select

from governance_engine.db.models import (
    AgentRoleRecord,
    CycleEventRecord,
    PhaseRecord,
    RuleRecord,
)
from governance_engine.interfaces import (
    DomainAdapter,
    EnforcementResult,
    GovernanceAgent,
    Primitive,
)
from governance_engine.phase_registry import PhaseRegistry
from governance_engine.phase_templates import PHASE_TEMPLATE_REGISTRY
from governance_engine.primitives import PRIMITIVE_REGISTRY
from governance_engine.role_stack import RoleStack
from governance_engine.utils import strip_json_fences

_KNOWN_RULE_CATEGORIES = {"LIMITS", "MONITORS", "SANCTIONS", "RESPONSIBLE", "CONDITIONAL"}


def _filter_rule_categories(raw_text: str, show_cats: dict[str, bool]) -> str:
    """Remove [CATEGORY] lines from a synthesized rule where that category is disabled.

    Lines that don't start with a known [CATEGORY] tag (e.g. Contributors) pass through unchanged.
    If show_cats is empty every line is kept as-is.
    """
    if not show_cats:
        return raw_text
    lines = []
    for line in raw_text.splitlines():
        stripped = line.strip()
        matched = next(
            (cat for cat in _KNOWN_RULE_CATEGORIES if stripped.startswith(f"[{cat}]")),
            None,
        )
        if matched is None or show_cats.get(matched.lower(), True):
            lines.append(line)
    return "\n".join(lines)


# Enforcement order per spec
_ENFORCEMENT_ORDER = [
    "cap",
    "declare",
    "monitor",
    "penalise",
    "adjust",
    "redistribute",
    "assign_role",
]


class GovernanceOrchestrator:
    def __init__(
        self,
        adapter: DomainAdapter,
        session: Session,
        llm_client,
        model_name: str,
        config: dict,
    ):
        self._adapter = adapter
        self._session = session
        self._llm_client = llm_client
        self._model = model_name
        self._config = config
        self._current_round: int = 0
        self._current_cycle: int | str = getattr(adapter, "_cycle_number", 1)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_cycle(self, round_number: int) -> dict:
        self._current_round = round_number
        self._current_cycle = getattr(self._adapter, "_cycle_number", self._current_cycle)
        _ctx = f"[cycle={self._current_cycle} | round={round_number}]"

        world_state = self._build_world_state(round_number)
        vocab = self._adapter.get_prompt_vocabulary()
        registry = PhaseRegistry(self._session, round_number)

        # Collapse check — inject recovery phases at the front
        if self._adapter.is_collapse_condition_met(world_state):
            logger.warning(f"{_ctx} Collapse condition met — injecting reflect+vote phases")
            world_state["collapse_triggered"] = True
            if "reflect_phase" not in registry.get_phase_order():
                registry.inject_phase("reflect_phase", "all", 0, None)
            if "vote_phase" not in registry.get_phase_order():
                registry.inject_phase("vote_phase", "all", 1, None)

        # Reconcile phases required by the active rule (idempotent — safe to call every round)
        active_rule = self._get_active_rule()
        if active_rule:
            registry.reconcile(active_rule, round_number)

        # Execute phases
        active_phases = registry.get_active_phases()
        phase_order = [p.name for p in active_phases]
        active_primitives_names = []
        if active_rule:
            try:
                pdicts = json.loads(active_rule.primitives_json or "[]")
                active_primitives_names = [d.get("primitive") for d in pdicts]
            except Exception:
                pass

        logger.bind(log_type="governance").info(
            f"{_ctx} ROUND START\n"
            f"  phases      : {phase_order or ['(none)']}\n"
            f"  primitives  : {active_primitives_names or ['(none)']}\n"
            f"  active_rule : {active_rule.id if active_rule else 'none'}\n"
            f"  stock       : {world_state.get('stock', '?'):.1f}\n"
            f"  agents      : {len(self._adapter.get_governing_agents())} governed"
            f" + {len(self._adapter.get_agents()) - len(self._adapter.get_governing_agents())} outsiders"
        )
        logger.info(f"{_ctx} phases={phase_order or ['(none)']}")

        for phase_record in active_phases:
            self._run_phase(phase_record, vocab, world_state, round_number)

        # Collect agent actions and run enforcement chain
        agent_actions: dict[str, dict] = world_state.get("pending_actions", {})
        if not agent_actions:
            agent_actions = self._collect_agent_actions(world_state)
            world_state["pending_actions"] = agent_actions

        active_primitives = self._load_active_primitives()
        agents_by_id = {a.id: a for a in self._adapter.get_agents()}
        governed_ids = {a.id for a in self._adapter.get_governing_agents()}

        enforce = self._config.get("enforce_rules", True)

        for agent_id, action in agent_actions.items():
            if agent_id in governed_ids and enforce:
                final_action = self._run_enforcement_chain(
                    agent_id, action, world_state, active_primitives, round_number
                )
            else:
                final_action = action  # outsiders always bypass; governed bypass when enforcement off
            result = self._adapter.resource.apply_action(agent_id, final_action["amount"])
            world_state.setdefault("agent_results", {})[agent_id] = result
            world_state.setdefault("agent_actions", {})[agent_id] = final_action["amount"]

            # Record harvest on agent for next-round history
            agent = agents_by_id.get(agent_id)
            if agent is not None and hasattr(agent, "record_harvest"):
                violations = world_state.get("pending_violations", {}).get(agent_id, [])
                cap_violations = [v for v in violations if v.get("trigger") == "exceed_cap"]
                capped = result["applied"] < action.get("amount", 0.0)

                if capped and cap_violations:
                    cap_info = world_state.get("cap_info", {})
                    cap_val = cap_info.get("cap_value", result["applied"])
                    basis = cap_info.get("basis", "fixed_units")
                    param_val = cap_info.get("param_value", cap_val)
                    requested = action.get("amount", 0.0)
                    # Human-readable cap description
                    if basis == "pct_of_stock":
                        cap_desc = f"{param_val * 100:.0f}% of stock ({cap_val:.0f} units)"
                    elif basis == "pct_of_total_catch":
                        cap_desc = f"{param_val * 100:.0f}% of total catch ({cap_val:.0f} units)"
                    elif basis == "sustainable_yield":
                        cap_desc = f"{param_val * 100:.0f}% of sustainable yield ({cap_val:.0f} units)"
                    else:
                        cap_desc = f"{cap_val:.0f} units"
                    has_redistribute = any(p.name == "redistribute" for p in active_primitives)
                    consequence = (
                        "excess redistributed to others per active rule"
                        if has_redistribute
                        else "excess returned to pool per active rule"
                    )
                    unusual = (
                        f"Tried to take {requested:.0f}; cap = {cap_desc}, "
                        f"actual take {result['applied']:.0f} — {consequence}"
                    )
                elif violations:
                    unusual = f"Violation: {violations[0].get('trigger', 'unknown')}"
                elif capped:
                    unusual = f"Enforcement adjusted take to {result['applied']:.1f}"
                else:
                    unusual = ""

                agent.record_harvest(
                    amount=result["applied"],
                    world_state=world_state,
                    observation=action.get("observation", ""),
                    unusual=unusual,
                )

        # Update public-takes ledger if a post_round / actual_take / public declare is active
        _public_declare = next(
            (
                p for p in active_primitives
                if p.name == "declare"
                and p.parameters.get("timing") == "post_round"
                and p.parameters.get("content") == "actual_take"
                and p.parameters.get("visibility") == "public"
            ),
            None,
        )
        if _public_declare:
            self._adapter._last_round_agent_takes = {
                aid: world_state["agent_actions"].get(aid, 0.0)
                for aid in governed_ids
            }

        # End-of-round adjust / redistribute passes
        self._run_end_of_round_primitives(active_primitives, world_state, round_number)

        # Expire roles (governed agents only)
        for agent in self._adapter.get_governing_agents():
            RoleStack(agent.id, self._session).pop_expired(round_number)

        self._adapter.on_cycle_complete(world_state)
        logger.info(f"Round {round_number}: cycle complete", extra={"world_state": json.dumps(world_state, default=str)})
        return world_state

    # ------------------------------------------------------------------
    # Phase execution
    # ------------------------------------------------------------------

    def _run_phase(
        self,
        phase_record: PhaseRecord,
        vocab: dict,
        world_state: dict,
        round_number: int,
    ) -> None:
        template_cls = PHASE_TEMPLATE_REGISTRY.get(phase_record.template_name)
        if template_cls is None:
            logger.warning(f"No template registered for '{phase_record.template_name}'")
            return

        active_rule = self._get_active_rule()
        primitive_params = self._get_phase_primitive_params(phase_record.template_name, active_rule)

        # ElectPhaseTemplate needs a session to write AgentRoleRecord
        if phase_record.template_name == "elect_phase":
            template = template_cls(vocab, primitive_params, session=self._session)
        else:
            template = template_cls(vocab, primitive_params)

        targets = self._resolve_targets(template.target, world_state)
        logger.bind(log_type="governance").info(
            f"[cycle={self._current_cycle} | round={round_number} | phase={phase_record.name}]"
            f" target='{template.target}' collect='{template.collect}'"
            f" agents={[a.id for a in targets]}"
        )
        responses: dict[str, str] = {}

        for agent in targets:
            role_overlay = RoleStack(agent.id, self._session).build_role_overlay(vocab)
            prompt = self._assemble_prompt(agent, phase_record, template, world_state, role_overlay)
            try:
                response = agent.receive_prompt(prompt)
            except Exception as e:
                logger.error(f"Agent {agent.id} failed to respond in phase {phase_record.name}: {e}")
                response = ""
            responses[agent.id] = response

        events = template.on_complete(responses, world_state)

        for event in events:
            self._apply_event(event, world_state)
            self._write_cycle_event(
                round_number=round_number,
                phase_name=phase_record.name,
                agent_id=event.get("agent_id", "system"),
                event_type=event.get("type", "unknown"),
                payload=event,
                source_rule_id=phase_record.source_rule_id,
            )

    # ------------------------------------------------------------------
    # Prompt assembly
    # ------------------------------------------------------------------

    def _assemble_prompt(
        self,
        agent: GovernanceAgent,
        phase: PhaseRecord,
        template,
        world_state: dict,
        role_overlay: str = "",
    ) -> str:
        base_ctx = agent.get_base_context()
        base_section = "\n".join(f"{k}: {v}" for k, v in base_ctx.items())

        phase_prompt = template.render_prompt(world_state)

        parts = [base_section]
        if role_overlay:
            parts.append(role_overlay)
        parts.append(phase_prompt)

        return "\n\n".join(parts)

    # ------------------------------------------------------------------
    # Enforcement chain
    # ------------------------------------------------------------------

    def _run_enforcement_chain(
        self,
        agent_id: str,
        action: dict,
        world_state: dict,
        primitives: list[Primitive],
        round_number: int,
    ) -> dict:
        ordered = self._order_primitives(primitives)
        current_action = dict(action)

        for primitive in ordered:
            result: EnforcementResult = primitive.enforce(agent_id, current_action, world_state)

            for event in result.events:
                event.setdefault("agent_id", agent_id)
                self._apply_event(event, world_state)
                active_rule = self._get_active_rule()
                self._write_cycle_event(
                    round_number=round_number,
                    phase_name=f"enforcement:{primitive.name}",
                    agent_id=agent_id,
                    event_type=event.get("type", "enforcement_event"),
                    payload=event,
                    source_rule_id=active_rule.id if active_rule else None,
                )

            if not result.permitted or result.adjusted_amount != current_action.get("amount"):
                original = current_action.get("amount", 0.0)
                current_action["amount"] = result.adjusted_amount
                logger.bind(log_type="governance").info(
                    f"[cycle={self._current_cycle} | round={round_number}"
                    f" | enforcement={primitive.name}]"
                    f" agent={agent_id}"
                    f" {original:.1f} → {result.adjusted_amount:.1f}"
                    f" permitted={result.permitted}"
                    f" violations={[v.get('type') for v in result.violations]}"
                )

        return current_action

    def _run_end_of_round_primitives(
        self,
        primitives: list[Primitive],
        world_state: dict,
        round_number: int,
    ) -> None:
        for primitive in primitives:
            if primitive.name in ("adjust", "redistribute"):
                synthetic_action = {"amount": 0.0, "end_of_round": True}
                for agent in self._adapter.get_governing_agents():
                    result = primitive.enforce(agent.id, synthetic_action, world_state)
                    for event in result.events:
                        event.setdefault("agent_id", agent.id)
                        self._apply_event(event, world_state)

    def _order_primitives(self, primitives: list[Primitive]) -> list[Primitive]:
        order = {name: idx for idx, name in enumerate(_ENFORCEMENT_ORDER)}
        return sorted(primitives, key=lambda p: order.get(p.name, 99))

    # ------------------------------------------------------------------
    # Event application
    # ------------------------------------------------------------------

    _STRUCTURAL_EVENTS = frozenset({
        "role_assigned", "quota_adjusted", "penalty_applied",
        "redistribution_applied", "vote_complete", "monitor_violation_recorded",
        "cap_violation", "declare_violation", "role_obligation_unmet",
    })

    def _apply_event(self, event: dict, world_state: dict) -> None:
        etype = event.get("type", "")
        agent_id = event.get("agent_id", "")

        if etype in self._STRUCTURAL_EVENTS:
            logger.bind(log_type="governance").info(
                f"[cycle={self._current_cycle} | round={self._current_round}"
                f" | event={etype}]"
                f" {json.dumps({k: v for k, v in event.items() if k not in ('phase', 'primitive')}, default=str)}"
            )

        if etype == "declaration_recorded":
            world_state.setdefault("declarations", {})[agent_id] = {
                "intended_amount": event.get("intended_amount", 0.0)
            }

        elif etype == "role_assigned":
            world_state.setdefault("role_assignments", {})[event.get("role", "")] = agent_id

        elif etype == "penalty_applied":
            amount = event.get("amount", 0.0)
            dest = event.get("destination", "pool")
            if dest == "pool":
                world_state["stock"] = world_state.get("stock", 0.0) + amount
            elif dest == "communal_fund":
                world_state["communal_fund"] = world_state.get("communal_fund", 0.0) + amount

        elif etype == "redistribution_applied":
            deltas: dict = event.get("per_agent_deltas", {})
            agent_balances: dict = world_state.setdefault("agent_balances", {})
            for aid, delta in deltas.items():
                agent_balances[aid] = agent_balances.get(aid, 0.0) + delta

        elif etype == "quota_adjusted":
            current = world_state.get("current_quota", float("inf"))
            direction = event.get("direction", "reduce")
            value = event.get("value", 0.0)
            if direction == "reduce":
                world_state["current_quota"] = max(0.0, current - value)
            elif direction == "restore":
                world_state["current_quota"] = current + value
            elif direction == "recalculate":
                world_state["current_quota"] = value

        elif etype == "vote_complete":
            winner = event.get("winner")
            if winner:
                world_state["winning_proposal"] = winner

        elif etype == "cap_violation":
            pending = world_state.setdefault("pending_violations", {})
            pending.setdefault(agent_id, []).append({
                "trigger": "exceed_cap",
                "excess": event.get("excess", 0.0),
            })

        elif etype == "declare_violation":
            pending = world_state.setdefault("pending_violations", {})
            pending.setdefault(agent_id, []).append({"trigger": "fail_declare"})

        else:
            logger.debug(f"Unhandled event type '{etype}' — stored in world_state events log")
            world_state.setdefault("event_log", []).append(event)

    # ------------------------------------------------------------------
    # Target resolution
    # ------------------------------------------------------------------

    def _resolve_targets(self, target_str: str, world_state: dict) -> list[GovernanceAgent]:
        governed = self._adapter.get_governing_agents()

        if target_str == "all":
            return governed

        if target_str.startswith("role:"):
            role = target_str.split(":", 1)[1]
            assigned_id = world_state.get("role_assignments", {}).get(role)
            if assigned_id is None:
                stmt = (
                    select(AgentRoleRecord)
                    .where(AgentRoleRecord.role == role)
                    .where(AgentRoleRecord.expires_round > world_state.get("round", 0))
                )
                records = self._session.exec(stmt).all()
                if records:
                    assigned_id = records[0].agent_id
            if assigned_id:
                return [a for a in governed if a.id == assigned_id]
            return []

        if target_str.startswith("agent:"):
            target_id = target_str.split(":", 1)[1]
            return [a for a in governed if a.id == target_id]

        return governed

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_world_state(self, round_number: int) -> dict:
        resource_state = self._adapter.resource.get_state()
        state = dict(resource_state)
        state["round"] = round_number
        state["agent_ids"] = [a.id for a in self._adapter.get_agents()]
        state["n_governed_agents"] = len(self._adapter.get_governing_agents())
        state["declarations"] = {}
        state["pending_violations"] = {}
        state["role_assignments"] = {}
        state["agent_actions"] = {}
        state["sustainable_threshold"] = self._adapter.resource.get_sustainable_threshold()
        state["resource_state"] = resource_state

        # Cycle metadata for prompt rendering
        state["cycle_number"] = getattr(self._adapter, "_cycle_number", 1)
        state["previous_cycle_collapsed"] = getattr(self._adapter, "_previous_cycle_collapsed", False)
        state["previous_collapse_round"] = getattr(self._adapter, "_previous_collapse_round", None)

        # Agreed rules for prompt rendering (all rules ever created)
        state["agreed_rules"] = self._get_agreed_rules_for_prompt()

        # Replenishment level hint for agent prompts
        if hasattr(self._adapter.resource, "get_replenishment_level"):
            state["replenishment_level"] = self._adapter.resource.get_replenishment_level(round_number)

        # Published-takes ledger: only populated when a post_round/actual_take/public
        # declare primitive is active, so agents see last round's harvests.
        state["published_takes"] = {}
        _ar = self._get_active_rule()
        if _ar:
            try:
                _pdicts = json.loads(_ar.primitives_json or "[]")
                _has_public_declare = any(
                    pd.get("primitive") == "declare"
                    and pd.get("parameters", {}).get("timing") == "post_round"
                    and pd.get("parameters", {}).get("content") == "actual_take"
                    and pd.get("parameters", {}).get("visibility") == "public"
                    for pd in _pdicts
                )
                if _has_public_declare:
                    state["published_takes"] = getattr(
                        self._adapter, "_last_round_agent_takes", {}
                    )
            except Exception:
                pass

        # Load role assignments from DB
        stmt = (
            select(AgentRoleRecord)
            .where(AgentRoleRecord.expires_round > round_number)
        )
        records = self._session.exec(stmt).all()
        for r in records:
            state["role_assignments"][r.role] = r.agent_id

        return state

    def _collect_agent_actions(self, world_state: dict) -> dict[str, dict]:
        """Ask each agent for their action decision using the adapter's prompt builder."""
        actions: dict[str, dict] = {}

        for agent in self._adapter.get_agents():
            prompt = self._adapter.build_action_prompt(agent, world_state)
            try:
                raw = agent.receive_prompt(prompt)
                data = json.loads(strip_json_fences(raw))
                amount = float(data.get("take", data.get("amount", 0.0)))
                actions[agent.id] = {
                    "amount": amount,
                    "observation": data.get("observation", ""),
                    "reasoning": data.get("reasoning", ""),
                    "rule_compliance": data.get("rule_compliance", ""),
                }
            except (json.JSONDecodeError, ValueError, TypeError):
                # Last resort: try to pull a bare number from the response
                try:
                    amount = float(str(raw).strip().split()[0])
                except Exception:
                    amount = 0.0
                actions[agent.id] = {"amount": amount, "observation": "", "reasoning": "", "rule_compliance": ""}
                logger.warning(f"Agent {agent.id}: non-JSON response, defaulted to amount={amount}")

        return actions

    def _load_active_primitives(self) -> list[Primitive]:
        active_rule = self._get_active_rule()
        if active_rule is None:
            return []

        primitive_dicts: list[dict] = json.loads(active_rule.primitives_json or "[]")
        primitives: list[Primitive] = []

        for pdata in primitive_dicts:
            primitive_name = pdata.get("primitive", "")
            cls = PRIMITIVE_REGISTRY.get(primitive_name)
            if cls is None:
                continue
            try:
                primitives.append(cls(pdata.get("parameters", {})))
            except Exception as e:
                logger.warning(f"Failed to instantiate primitive {primitive_name}: {e}")

        return primitives

    def _get_active_rule(self) -> RuleRecord | None:
        stmt = select(RuleRecord).where(RuleRecord.status == "active")
        return self._session.exec(stmt).first()

    def _get_phase_primitive_params(
        self, template_name: str, active_rule: RuleRecord | None
    ) -> dict:
        if active_rule is None:
            return {}

        try:
            primitive_dicts: list[dict] = json.loads(active_rule.primitives_json or "[]")
        except (json.JSONDecodeError, TypeError):
            return {}

        for pdata in primitive_dicts:
            primitive_name = pdata.get("primitive", "")
            cls = PRIMITIVE_REGISTRY.get(primitive_name)
            if cls is None:
                continue
            try:
                instance = cls(pdata.get("parameters", {}))
                if template_name in instance.required_phases():
                    return pdata.get("parameters", {})
            except Exception:
                continue

        return {}

    def _get_agreed_rules_for_prompt(self) -> list[dict]:
        stmt = select(RuleRecord).where(RuleRecord.status == "active").order_by(RuleRecord.id)
        rules = self._session.exec(stmt).all()
        show_cats = self._config.get("show_rule_categories", {})
        return [
            {"number": f"R{i + 1}", "text": _filter_rule_categories(r.raw_text, show_cats)}
            for i, r in enumerate(rules)
        ]

    def _write_cycle_event(
        self,
        round_number: int,
        phase_name: str,
        agent_id: str,
        event_type: str,
        payload: dict,
        source_rule_id: int | None,
    ) -> None:
        record = CycleEventRecord(
            round_number=round_number,
            phase_name=phase_name,
            agent_id=agent_id,
            event_type=event_type,
            payload_json=json.dumps(payload, default=str),
            source_rule_id=source_rule_id,
            timestamp=datetime.utcnow(),
        )
        self._session.add(record)
        self._session.commit()
