from __future__ import annotations

from sqlmodel import Session, select

from governance_engine.db.models import AgentRoleRecord


class RoleStack:
    def __init__(self, agent_id: str, session: Session):
        self._agent_id = agent_id
        self._session = session

    def get_active_roles(self, current_round: int) -> list[str]:
        stmt = (
            select(AgentRoleRecord)
            .where(AgentRoleRecord.agent_id == self._agent_id)
            .where(AgentRoleRecord.expires_round > current_round)
        )
        records = self._session.exec(stmt).all()
        return [r.role for r in records]

    def push_role(self, role: str, expires_round: int, source_rule_id: int) -> None:
        record = AgentRoleRecord(
            agent_id=self._agent_id,
            role=role,
            assigned_round=0,
            expires_round=expires_round,
            source_rule_id=source_rule_id,
        )
        self._session.add(record)
        self._session.commit()

    def pop_expired(self, current_round: int) -> None:
        stmt = (
            select(AgentRoleRecord)
            .where(AgentRoleRecord.agent_id == self._agent_id)
            .where(AgentRoleRecord.expires_round <= current_round)
        )
        expired = self._session.exec(stmt).all()
        for record in expired:
            self._session.delete(record)
        if expired:
            self._session.commit()

    def build_role_overlay(self, vocabulary: dict) -> str:
        # Fetch current round from outside — caller passes roles directly
        stmt = select(AgentRoleRecord).where(AgentRoleRecord.agent_id == self._agent_id)
        records = self._session.exec(stmt).all()
        if not records:
            return ""

        parts = ["Your active roles this round:"]
        for r in records:
            obligation_hint = vocabulary.get(r.role, r.role)
            parts.append(f"  - {r.role}: {obligation_hint} (active until round {r.expires_round})")

        return "\n".join(parts)
