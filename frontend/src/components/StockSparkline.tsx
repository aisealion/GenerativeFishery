import { Line, LineChart, ResponsiveContainer, Tooltip, YAxis } from "recharts";

export interface StockPoint {
  round: number;
  stock: number;
}

export function StockSparkline({
  history,
  carryingCapacity,
}: {
  history: StockPoint[];
  carryingCapacity: number;
}) {
  return (
    <div style={{ width: "100%", height: 60 }}>
      <ResponsiveContainer>
        <LineChart data={history}>
          <YAxis domain={[0, carryingCapacity]} hide />
          <Tooltip
            formatter={(value) => Number(value).toFixed(1)}
            labelFormatter={(round) => `round ${round}`}
          />
          <Line type="monotone" dataKey="stock" stroke="#2563eb" dot={false} strokeWidth={2} />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
