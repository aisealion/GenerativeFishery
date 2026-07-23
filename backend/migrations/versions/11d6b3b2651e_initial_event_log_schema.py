"""initial event log schema

Revision ID: 11d6b3b2651e
Revises:
Create Date: 2026-07-11 17:40:17.254270

"""
from typing import Sequence, Union

from alembic import op


# revision identifiers, used by Alembic.
revision: str = '11d6b3b2651e'
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS timescaledb")

    op.execute(
        """
        CREATE TABLE events (
            id BIGSERIAL,
            fishery_id TEXT NOT NULL,
            round INT NOT NULL,
            phase TEXT NOT NULL,
            type TEXT NOT NULL,
            actor_id TEXT,
            target_id TEXT,
            visibility TEXT NOT NULL,
            payload JSONB NOT NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            PRIMARY KEY (id, created_at)
        )
        """
    )
    # create_hypertable requires the partitioning column in the PK/unique
    # constraints (hence the composite PK above rather than id alone).
    op.execute("SELECT create_hypertable('events', 'created_at')")

    op.execute("CREATE INDEX ix_events_fishery_round ON events (fishery_id, round)")
    op.execute("CREATE INDEX ix_events_actor ON events (actor_id)")
    op.execute("CREATE INDEX ix_events_target ON events (target_id)")

    # LISTEN/NOTIFY fan-out: NOTIFY fires in the same transaction as the
    # insert, on channel 'events'. The payload column is deliberately
    # excluded from the notification body -- NOTIFY payloads are capped at
    # 8000 bytes by Postgres and event payloads have no size bound, so
    # listeners get the routing/identity fields here and re-fetch full rows
    # (e.g. `GET /fisheries/{id}/state` or a follow-up SELECT) when they
    # need `payload` itself.
    op.execute(
        """
        CREATE FUNCTION notify_event() RETURNS trigger AS $$
        BEGIN
            PERFORM pg_notify(
                'events',
                json_build_object(
                    'id', NEW.id,
                    'fishery_id', NEW.fishery_id,
                    'round', NEW.round,
                    'phase', NEW.phase,
                    'type', NEW.type,
                    'actor_id', NEW.actor_id,
                    'target_id', NEW.target_id,
                    'visibility', NEW.visibility
                )::text
            );
            RETURN NEW;
        END;
        $$ LANGUAGE plpgsql
        """
    )
    op.execute(
        """
        CREATE TRIGGER events_notify
        AFTER INSERT ON events
        FOR EACH ROW EXECUTE FUNCTION notify_event()
        """
    )


def downgrade() -> None:
    op.execute("DROP TRIGGER IF EXISTS events_notify ON events")
    op.execute("DROP FUNCTION IF EXISTS notify_event")
    op.execute("DROP TABLE IF EXISTS events")
