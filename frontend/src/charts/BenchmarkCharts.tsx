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
  random: "#64748B",
  least_confidence: "#D97706",
  margin: "#7C3AED",
  entropy: "#159C91",
  bald: "#DB2777",
  coreset: "#059669",
  badge: "#718BE0",
  hybrid: "#4568D9",
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
          <CartesianGrid strokeDasharray="3 3" stroke="#D5DDF2" vertical={false} />
          <XAxis
            dataKey="strategy"
            tick={{ fontSize: 11, fill: "#64748B" }}
            axisLine={{ stroke: "#C7D3F0" }}
            tickLine={{ stroke: "#C7D3F0" }}
            angle={-15}
            textAnchor="end"
            height={48}
          />
          <YAxis
            tickFormatter={(v) => `${v}%`}
            tick={{ fontSize: 11, fill: "#64748B" }}
            axisLine={{ stroke: "#C7D3F0" }}
            tickLine={{ stroke: "#C7D3F0" }}
          />
          <Tooltip
            contentStyle={{
              backgroundColor: "#F3F5FD",
              borderColor: "#C7D3F0",
              borderRadius: "10px",
              fontSize: "12px",
              color: "#263452",
            }}
            formatter={(val: number, name: string) => [
              `${val}%`,
              name === "alc" ? "ALC (×100)" : "Label Reduction vs Random",
            ]}
          />
          <Legend wrapperStyle={{ fontSize: "12px" }} />
          <Bar dataKey="alc" name="ALC Score (%)" fill="#4568D9" radius={[6, 6, 0, 0]} />
          <Bar
            dataKey="reduction"
            name="Label Reduction vs Random (%)"
            fill="#159C91"
            radius={[6, 6, 0, 0]}
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
      <div className="flex h-[320px] items-center justify-center rounded-xl border border-dashed border-[#C7D3F0] bg-[#EEF1FB]/80 text-xs text-[#64748B]">
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
          <CartesianGrid strokeDasharray="3 3" stroke="#D5DDF2" />
          <XAxis
            dataKey="n_labeled"
            tick={{ fontSize: 12, fill: "#64748B" }}
            axisLine={{ stroke: "#C7D3F0" }}
            tickLine={{ stroke: "#C7D3F0" }}
            label={{
              value: "Annotation Budget (Labeled Samples)",
              position: "insideBottom",
              offset: -6,
              style: { fontSize: 11, fill: "#64748B", fontWeight: 500 },
            }}
          />
          <YAxis
            tickFormatter={(v) => `${v}%`}
            tick={{ fontSize: 12, fill: "#64748B" }}
            axisLine={{ stroke: "#C7D3F0" }}
            tickLine={{ stroke: "#C7D3F0" }}
          />
          <Tooltip
            contentStyle={{
              backgroundColor: "#F3F5FD",
              borderColor: "#C7D3F0",
              borderRadius: "10px",
              fontSize: "12px",
              color: "#263452",
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
              stroke={STRATEGY_COLORS[strat] || "#4568D9"}
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

