import type { NormSnapshot } from "../api";

export function ActiveNorms({ norms, groupNormText }: { norms: NormSnapshot[]; groupNormText: string }) {
  return (
    <div className="active-norms">
      <p className="group-norm">
        <strong>Community policy:</strong> {groupNormText}
      </p>
      <ul>
        {norms.map((norm) => (
          <li key={norm.id}>
            <span className="norm-type">{norm.type}</span>
            {norm.source_text ? ` — "${norm.source_text}"` : " (default)"}
          </li>
        ))}
      </ul>
    </div>
  );
}
