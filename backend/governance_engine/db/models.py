from __future__ import annotations

import os
from datetime import datetime
from typing import Optional

from sqlmodel import Field, Session, SQLModel, create_engine


class RuleRecord(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    raw_text: str
    primitives_json: str
    vote_count: int = 0
    status: str = "proposed"  # proposed | active | expired
    created_round: int
    activated_round: Optional[int] = None
    expired_round: Optional[int] = None


class PhaseRecord(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    name: str
    order_index: int
    active: bool = True
    source_rule_id: Optional[int] = Field(default=None, foreign_key="rulerecord.id")
    template_name: str
    target: str
    round_injected: int
    expires_round: Optional[int] = None


class AgentRoleRecord(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    agent_id: str
    role: str
    assigned_round: int
    expires_round: int
    source_rule_id: int


class CycleEventRecord(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    round_number: int
    phase_name: str
    agent_id: str
    event_type: str
    payload_json: str
    source_rule_id: Optional[int] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


_DB_PATH = os.environ.get("GOVERNANCE_DB_PATH", "governance.db")
_engine = None


def get_engine():
    global _engine
    if _engine is None:
        _engine = create_engine(f"sqlite:///{_DB_PATH}", echo=False)
    return _engine


def init_db():
    SQLModel.metadata.create_all(get_engine())


def get_session():
    return Session(get_engine())
