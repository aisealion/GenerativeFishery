"""SQLAlchemy Core table definitions mirroring the hand-written schema.

Kept as plain Core `Table` objects (not an ORM model) since the schema is
owned by the Alembic migrations in `migrations/versions/`, including
TimescaleDB-specific DDL (hypertable, trigger) that has no SQLAlchemy
metadata equivalent. This is only used to build typed insert/select
statements against the existing table.
"""

from sqlalchemy import BigInteger, Column, DateTime, Integer, MetaData, Table, Text, func
from sqlalchemy.dialects.postgresql import JSONB

metadata = MetaData()

events_table = Table(
    "events",
    metadata,
    # `id`/`created_at` are DB-generated (BIGSERIAL, DEFAULT now()) -- marked
    # here so SQLAlchemy's composite-PK metadata matches the real schema,
    # even though our inserts never provide them (Postgres fills both; we
    # read them back via `.returning(...)`).
    Column("id", BigInteger, primary_key=True, autoincrement=True),
    Column("fishery_id", Text, nullable=False),
    Column("round", Integer, nullable=False),
    Column("phase", Text, nullable=False),
    Column("type", Text, nullable=False),
    Column("actor_id", Text),
    Column("target_id", Text),
    Column("visibility", Text, nullable=False),
    Column("payload", JSONB, nullable=False),
    Column(
        "created_at",
        DateTime(timezone=True),
        primary_key=True,
        nullable=False,
        server_default=func.now(),
    ),
)
