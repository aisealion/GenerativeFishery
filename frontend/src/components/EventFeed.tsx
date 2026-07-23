import type { FisheryEventSummary } from "../api";

function describe(event: FisheryEventSummary): string {
  const who = [event.actor_id, event.target_id].filter(Boolean).join(" -> ");
  return who ? `${event.type} (${who})` : event.type;
}

export function EventFeed({ events }: { events: FisheryEventSummary[] }) {
  return (
    <ul className="event-feed">
      {events.map((event) => (
        <li key={event.id} className={event.type === "migration" ? "migration" : ""}>
          <span className="round-tag">r{event.round}</span>
          <span className="phase-tag">{event.phase}</span>
          {describe(event)}
        </li>
      ))}
    </ul>
  );
}
