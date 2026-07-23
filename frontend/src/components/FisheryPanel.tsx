import { useEffect, useRef, useState } from "react";
import { useFisheryLive } from "../api";
import { ActiveNorms } from "./ActiveNorms";
import { EventFeed } from "./EventFeed";
import { Roster } from "./Roster";
import { StockSparkline, type StockPoint } from "./StockSparkline";

const MAX_HISTORY = 40;

export function FisheryPanel({
  fisheryId,
  onMigration,
}: {
  fisheryId: string;
  onMigration?: (message: string) => void;
}) {
  const { state, events, connected, error } = useFisheryLive(fisheryId);
  const [stockHistory, setStockHistory] = useState<StockPoint[]>([]);
  const lastRoundRecorded = useRef<number | null>(null);
  const lastSeenMigrationEventId = useRef<number | null>(null);

  useEffect(() => {
    if (state === null) return;
    if (lastRoundRecorded.current === state.round) return;
    lastRoundRecorded.current = state.round;
    setStockHistory((prev) => [...prev, { round: state.round, stock: state.stock }].slice(-MAX_HISTORY));
  }, [state]);

  // Migration events carry no payload over the WS stream (NOTIFY payloads
  // are capped at 8000 bytes, see the initial DB migration), so departure
  // vs. arrival is inferred from whether the agent is currently in *this*
  // fishery's own roster snapshot -- present means they just arrived here.
  useEffect(() => {
    const latest = events[0];
    if (!onMigration || !latest || latest.type !== "migration") return;
    if (lastSeenMigrationEventId.current === latest.id) return;
    lastSeenMigrationEventId.current = latest.id;

    const isHere = state?.agents.some((a) => a.agent_id === latest.actor_id) ?? false;
    onMigration(
      isHere
        ? `🎣 ${latest.actor_id} arrived in ${fisheryId}, carrying their prior experience`
        : `👋 ${latest.actor_id} left ${fisheryId} after it collapsed`
    );
  }, [events, state, fisheryId, onMigration]);

  const currentPhase = events[0]?.phase ?? "—";

  return (
    <section className="fishery-panel">
      <header>
        <h2>{fisheryId}</h2>
        <span className={`connection-dot ${connected ? "connected" : "disconnected"}`} />
      </header>

      {error && <p className="error">{error}</p>}

      {state ? (
        <>
          <div className="summary-row">
            <div>
              <strong>Round</strong> {state.round}
            </div>
            <div>
              <strong>Phase</strong> {currentPhase}
            </div>
            <div>
              <strong>Stock</strong> {state.stock.toFixed(1)} / {state.carrying_capacity}
            </div>
            {state.collapsed && <div className="collapsed-tag">COLLAPSED</div>}
          </div>

          <StockSparkline history={stockHistory} carryingCapacity={state.carrying_capacity} />

          <h3>Roster</h3>
          <Roster agents={state.agents} roles={state.roles} />

          <h3>Active norms</h3>
          <ActiveNorms norms={state.active_norms} groupNormText={state.group_norm_text} />

          <h3>Event feed</h3>
          <EventFeed events={events} />
        </>
      ) : (
        <p>Loading fishery state…</p>
      )}
    </section>
  );
}
