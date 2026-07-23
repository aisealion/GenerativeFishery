import { useEffect, useRef, useState } from "react";

export interface AgentSnapshot {
  agent_id: string;
  alive: boolean;
  payoff: number;
  last_effort: number | null;
  last_harvest: number | null;
  personal_norm: string | null;
  persona_type: "altruistic" | "selfish" | null;
  persona_description: string | null;
}

export interface NormSnapshot {
  id: string;
  type: string;
  source_text: string | null;
  detail: Record<string, unknown>;
}

export interface FisheryStateSnapshot {
  fishery_id: string;
  round: number;
  stock: number;
  carrying_capacity: number;
  collapsed: boolean;
  group_norm_text: string;
  roles: Record<string, string>;
  agents: AgentSnapshot[];
  active_norms: NormSnapshot[];
}

// The WS stream carries only lightweight event metadata (no `payload` --
// NOTIFY payloads are capped at 8000 bytes, see the initial migration), so
// the feed shows what/who/when but not full event detail.
export interface FisheryEventSummary {
  id: number;
  fishery_id: string;
  round: number;
  phase: string;
  type: string;
  actor_id: string | null;
  target_id: string | null;
  visibility: string;
}

const API_BASE = (import.meta.env.VITE_API_BASE as string | undefined) ?? "http://localhost:8000";
const WS_BASE = API_BASE.replace(/^http/, "ws");

export async function fetchFisheryState(fisheryId: string): Promise<FisheryStateSnapshot> {
  const response = await fetch(`${API_BASE}/fisheries/${fisheryId}/state`);
  if (!response.ok) {
    throw new Error(`GET /fisheries/${fisheryId}/state failed: ${response.status}`);
  }
  return (await response.json()) as FisheryStateSnapshot;
}

const MAX_EVENTS = 60;
const STATE_POLL_MS = 2000;

export function useFisheryLive(fisheryId: string): {
  state: FisheryStateSnapshot | null;
  events: FisheryEventSummary[];
  connected: boolean;
  error: string | null;
} {
  const [state, setState] = useState<FisheryStateSnapshot | null>(null);
  const [events, setEvents] = useState<FisheryEventSummary[]>([]);
  const [connected, setConnected] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const socketRef = useRef<WebSocket | null>(null);

  useEffect(() => {
    let cancelled = false;

    const refreshState = () => {
      fetchFisheryState(fisheryId)
        .then((snapshot) => {
          if (!cancelled) setState(snapshot);
        })
        .catch((err: Error) => {
          if (!cancelled) setError(err.message);
        });
    };

    refreshState();
    const pollId = window.setInterval(refreshState, STATE_POLL_MS);

    const socket = new WebSocket(`${WS_BASE}/fisheries/${fisheryId}/stream`);
    socketRef.current = socket;
    socket.onopen = () => {
      setConnected(true);
      // Clears a stale error from an earlier socket -- e.g. React StrictMode's
      // dev-mode double-invoke opens and immediately discards a first socket,
      // whose onerror shouldn't linger once a real connection succeeds.
      setError(null);
    };
    socket.onclose = () => setConnected(false);
    socket.onerror = () => setError("WebSocket connection error");
    socket.onmessage = (message) => {
      const event = JSON.parse(message.data as string) as FisheryEventSummary;
      setEvents((prev) => [event, ...prev].slice(0, MAX_EVENTS));
    };

    return () => {
      cancelled = true;
      window.clearInterval(pollId);
      socket.close();
    };
  }, [fisheryId]);

  return { state, events, connected, error };
}
