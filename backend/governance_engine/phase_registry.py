from __future__ import annotations

from loguru import logger
from sqlmodel import Session, select

from governance_engine.db.models import PhaseRecord, RuleRecord


class PhaseRegistry:
    def __init__(self, session: Session, round_number: int):
        self._session = session
        self._round = round_number

    def get_active_phases(self) -> list[PhaseRecord]:
        stmt = (
            select(PhaseRecord)
            .where(PhaseRecord.active == True)  # noqa: E712
            .order_by(PhaseRecord.order_index)
        )
        phases = self._session.exec(stmt).all()
        return [
            p for p in phases
            if p.expires_round is None or p.expires_round > self._round
        ]

    def inject_phase(
        self,
        template_name: str,
        target: str,
        order_index: int,
        source_rule_id: int | None,
        expires_round: int | None = None,
    ) -> PhaseRecord:
        # Shift existing phases with order_index >= insertion point
        existing = self._session.exec(
            select(PhaseRecord)
            .where(PhaseRecord.active == True)  # noqa: E712
            .where(PhaseRecord.order_index >= order_index)
        ).all()
        for p in existing:
            p.order_index += 1
            self._session.add(p)

        record = PhaseRecord(
            name=template_name,
            order_index=order_index,
            active=True,
            source_rule_id=source_rule_id,
            template_name=template_name,
            target=target,
            round_injected=self._round,
            expires_round=expires_round,
        )
        self._session.add(record)
        self._session.commit()
        self._session.refresh(record)

        logger.bind(log_type="governance").info(
            f"[round={self._round} | registry] INJECT phase='{template_name}'"
            f" index={order_index} target='{target}'"
            f" source_rule={source_rule_id} expires={expires_round}"
        )
        return record

    def deactivate_phase(self, phase_id: int) -> None:
        phase = self._session.get(PhaseRecord, phase_id)
        if phase:
            logger.bind(log_type="governance").info(
                f"[round={self._round} | registry] DEACTIVATE phase_id={phase_id}"
                f" name='{phase.name}' source_rule={phase.source_rule_id}"
            )
            phase.active = False
            self._session.add(phase)
            self._session.commit()

    def reconcile(self, active_rule: RuleRecord, current_round: int) -> None:
        """Idempotent: inject phases required by active_rule that aren't present; expire stale ones."""
        import json

        primitives_data: list[dict] = json.loads(active_rule.primitives_json or "[]")
        required_phases: set[str] = set()

        from governance_engine.primitives import PRIMITIVE_REGISTRY

        for pdata in primitives_data:
            primitive_name = pdata.get("primitive")
            cls = PRIMITIVE_REGISTRY.get(primitive_name)
            if cls is None:
                continue
            try:
                instance = cls(pdata.get("parameters", {}))
                required_phases.update(instance.required_phases())
            except Exception:
                continue

        active_phases = self.get_active_phases()
        active_template_names = {p.template_name for p in active_phases}

        logger.bind(log_type="governance").info(
            f"[round={current_round} | registry] RECONCILE rule_id={active_rule.id}"
            f" required={sorted(required_phases)} present={sorted(active_template_names)}"
        )

        for idx, phase_name in enumerate(sorted(required_phases)):
            if phase_name not in active_template_names:
                self.inject_phase(
                    template_name=phase_name,
                    target="all",
                    order_index=idx + 10,
                    source_rule_id=active_rule.id,
                    expires_round=None,
                )

        # Deactivate phases from a different rule that are no longer required
        for phase in active_phases:
            if (
                phase.source_rule_id is not None
                and phase.source_rule_id != active_rule.id
                and phase.template_name not in required_phases
            ):
                self.deactivate_phase(phase.id)

    def get_phase_order(self) -> list[str]:
        return [p.name for p in self.get_active_phases()]
