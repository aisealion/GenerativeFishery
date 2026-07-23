import type { AgentSnapshot } from "../api";

function personaBadge(personaType: AgentSnapshot["persona_type"]) {
  if (personaType === null) return null;
  return <span className={`persona-badge ${personaType}`}>{personaType === "altruistic" ? "A" : "S"}</span>;
}

export function Roster({ agents, roles }: { agents: AgentSnapshot[]; roles: Record<string, string> }) {
  const roleByAgent = new Map<string, string[]>();
  for (const [role, agentId] of Object.entries(roles)) {
    roleByAgent.set(agentId, [...(roleByAgent.get(agentId) ?? []), role]);
  }

  return (
    <table className="roster">
      <thead>
        <tr>
          <th>Agent</th>
          <th>Effort</th>
          <th>Last catch</th>
          <th>Payoff</th>
          <th>Role</th>
        </tr>
      </thead>
      <tbody>
        {agents.map((agent) => (
          <tr key={agent.agent_id} className={agent.alive ? "" : "dead"}>
            <td>
              {agent.agent_id} {personaBadge(agent.persona_type)}
            </td>
            <td>{agent.last_effort?.toFixed(2) ?? "—"}</td>
            <td>{agent.last_harvest?.toFixed(2) ?? "—"}</td>
            <td>{agent.payoff.toFixed(2)}</td>
            <td>{roleByAgent.get(agent.agent_id)?.join(", ") ?? ""}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}
