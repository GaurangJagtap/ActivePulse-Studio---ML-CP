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
            tick={{ fontSize: 11, fill: "#806D60" }}
          />
          <YAxis
            type="category"
            dataKey="name"
            width={85}
            tick={{ fontSize: 11, fill: "#263452" }}
          />
          <Tooltip
            contentStyle={{
              backgroundColor: "#FFFCF7",
              borderColor: "#E6D6C6",
              borderRadius: "8px",
              fontSize: "12px",
              color: "#38271F",
              boxShadow: "0 4px 12px rgba(56, 39, 31, 0.1)",
            }}
            formatter={(val: number) => [`${val}%`, "Probability"]}
          />
          <Bar dataKey="probability" radius={[0, 4, 4, 0]}>
            {data.map((entry) => (
              <Cell
                key={entry.name}
                fill={
                  entry.name === predictedClassName ? "#6B3E26" : "#E9D5C0"
                }
              />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

