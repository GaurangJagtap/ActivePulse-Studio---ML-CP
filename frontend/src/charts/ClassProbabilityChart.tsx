import {
  Bar,
  BarChart,
  Cell,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

interface ClassProbabilityChartProps {
  classNames: string[];
  probabilities: number[];
  predictedClassName?: string | null;
  height?: number;
}

export function ClassProbabilityChart({
  classNames,
  probabilities,
  predictedClassName,
  height = 220,
}: ClassProbabilityChartProps) {
  const data = classNames.map((name, i) => ({
    name,
    probability: Number(((probabilities[i] || 0) * 100).toFixed(1)),
  }));

  return (
    <div style={{ width: "100%", height }}>
      <ResponsiveContainer width="100%" height="100%">
        <BarChart
          data={data}
          layout="vertical"
          margin={{ top: 4, right: 16, left: 10, bottom: 4 }}
        >
          <XAxis
            type="number"
            domain={[0, 100]}
            tickFormatter={(v) => `${v}%`}
            tick={{ fontSize: 11, fill: "#64748B" }}
          />
          <YAxis
            type="category"
            dataKey="name"
            width={85}
            tick={{ fontSize: 11, fill: "#263452" }}
          />
          <Tooltip
            contentStyle={{
              backgroundColor: "#F3F5FD",
              borderColor: "#C7D3F0",
              borderRadius: "10px",
              fontSize: "12px",
              color: "#263452",
            }}
            formatter={(val: number) => [`${val}%`, "Probability"]}
          />
          <Bar dataKey="probability" radius={[0, 6, 6, 0]}>
            {data.map((entry) => (
              <Cell
                key={entry.name}
                fill={
                  entry.name === predictedClassName ? "#4568D9" : "#9AAAD0"
                }
              />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

