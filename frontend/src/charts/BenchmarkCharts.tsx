import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { BenchmarkCurvePoint, BenchmarkSavingsRow } from "@/types/api";

const STRATEGY_COLORS: Record<string, string> = {
  random: "#806D60",
  least_confidence: "#D99127",
  margin: "#6B3E26",
  entropy: "#B98255",
  bald: "#A87042",
  coreset: "#2D6A4F",
  badge: "#4E2B1A",
  hybrid: "#8C5832",
};

export function BenchmarkAlcBarChart({ rows }: { rows: BenchmarkSavingsRow[] }) {
  const data = rows.map((r) => ({
    strategy: r.strategy,
    alc: Number((r.alc_mean * 100).toFixed(2)),
    reduction: r.label_reduction_pct ?? 0,
  }));

  return (
    <div style={{ width: "100%", height: 320 }}>
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={data} margin={{ top: 10, right: 20, left: 0, bottom: 20 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#EFE4D8" vertical={false} />
          <XAxis
            dataKey="strategy"
            tick={{ fontSize: 11, fill: "#806D60" }}
            axisLine={{ stroke: "#E6D6C6" }}
            tickLine={{ stroke: "#E6D6C6" }}
            angle={-15}
            textAnchor="end"
            height={48}
          />
          <YAxis
            tickFormatter={(v) => `${v}%`}
            tick={{ fontSize: 11, fill: "#806D60" }}
            axisLine={{ stroke: "#E6D6C6" }}
            tickLine={{ stroke: "#E6D6C6" }}
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
            formatter={(val: number, name: string) => [
              `${val}%`,
              name === "alc" ? "ALC (×100)" : "Label Reduction vs Random",
            ]}
          />
          <Legend wrapperStyle={{ fontSize: "12px" }} />
          <Bar dataKey="alc" name="ALC Score (%)" fill="#6B3E26" radius={[4, 4, 0, 0]} />
          <Bar
            dataKey="reduction"
            name="Label Reduction vs Random (%)"
            fill="#B98255"
            radius={[4, 4, 0, 0]}
          />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

export function BenchmarkCurvesChart({
  points,
  height = 360,
}: {
  points: BenchmarkCurvePoint[];
  height?: number;
}) {
  if (!points.length) {
    return (
      <div className="flex h-[320px] items-center justify-center rounded-xl border border-dashed border-[#E6D6C6] bg-[#FFFCF7] text-xs text-[#806D60]">
        Run a multi-seed benchmark above to generate interactive per-round learning curves across strategies.
      </div>
    );
  }

  const strategies = Array.from(new Set(points.map((p) => p.strategy)));
  const groupedByN: Record<number, Record<string, number>> = {};

  for (const pt of points) {
    if (!groupedByN[pt.n_labeled]) {
      groupedByN[pt.n_labeled] = { n_labeled: pt.n_labeled };
    }
    groupedByN[pt.n_labeled][pt.strategy] = Number((pt.mean_acc * 100).toFixed(2));
  }

  const chartData = Object.values(groupedByN).sort(
    (a, b) => a.n_labeled - b.n_labeled
  );

  return (
    <div style={{ width: "100%", height }}>
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={chartData} margin={{ top: 12, right: 24, left: 4, bottom: 14 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#EFE4D8" />
          <XAxis
            dataKey="n_labeled"
            tick={{ fontSize: 12, fill: "#806D60" }}
            axisLine={{ stroke: "#E6D6C6" }}
            tickLine={{ stroke: "#E6D6C6" }}
            label={{
              value: "Annotation Budget (Labeled Samples)",
              position: "insideBottom",
              offset: -6,
              style: { fontSize: 11, fill: "#806D60", fontWeight: 500 },
            }}
          />
          <YAxis
            tickFormatter={(v) => `${v}%`}
            tick={{ fontSize: 12, fill: "#806D60" }}
            axisLine={{ stroke: "#E6D6C6" }}
            tickLine={{ stroke: "#E6D6C6" }}
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
            formatter={(val: number, name: string) => [`${val}%`, name]}
          />
          <Legend wrapperStyle={{ fontSize: "12px", paddingTop: "10px" }} />
          {strategies.map((strat) => (
            <Line
              key={strat}
              type="monotone"
              dataKey={strat}
              name={strat}
              stroke={STRATEGY_COLORS[strat] || "#6B3E26"}
              strokeWidth={strat === "random" ? 2 : 2.75}
              strokeDasharray={strat === "random" ? "5 5" : undefined}
              dot={{ r: 3.5 }}
            />
          ))}
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}

