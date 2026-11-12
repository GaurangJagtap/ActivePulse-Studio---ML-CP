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
  least_confidence: "#F59E0B",
  margin: "#8B5CF6",
  entropy: "#0D9488",
  bald: "#EC4899",
  coreset: "#16A34A",
  badge: "#0EA5E9",
  hybrid: "#2563EB",
};

export function BenchmarkAlcBarChart({ rows }: { rows: BenchmarkSavingsRow[] }) {
  const data = rows.map((r) => ({
    strategy: r.strategy,
    alc: Number((r.alc_mean * 100).toFixed(2)),
    reduction: r.label_reduction_pct ?? 0,
  }));

  return (
    <div style={{ width: "100%", height: 300 }}>
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={data} margin={{ top: 10, right: 20, left: 0, bottom: 20 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#E2E8F0" vertical={false} />
          <XAxis
            dataKey="strategy"
            tick={{ fontSize: 11, fill: "#64748B" }}
            angle={-15}
            textAnchor="end"
            height={48}
          />
          <YAxis tickFormatter={(v) => `${v}%`} tick={{ fontSize: 11, fill: "#64748B" }} />
          <Tooltip
            contentStyle={{
              backgroundColor: "#FFFFFF",
              borderColor: "#E2E8F0",
              borderRadius: "8px",
              fontSize: "12px",
            }}
            formatter={(val: number, name: string) => [
              `${val}%`,
              name === "alc" ? "ALC (×100)" : "Label Reduction vs Random",
            ]}
          />
          <Legend wrapperStyle={{ fontSize: "12px" }} />
          <Bar dataKey="alc" name="ALC Score (%)" fill="#2563EB" radius={[4, 4, 0, 0]} />
          <Bar
            dataKey="reduction"
            name="Label Reduction vs Random (%)"
            fill="#0D9488"
            radius={[4, 4, 0, 0]}
          />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

export function BenchmarkCurvesChart({
  points,
}: {
  points: BenchmarkCurvePoint[];
}) {
  if (!points.length) {
    return (
      <div className="flex h-[300px] items-center justify-center rounded-lg border border-dashed border-[#E2E8F0] bg-[#F8FAFC] text-xs text-[#64748B]">
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
    <div style={{ width: "100%", height: 320 }}>
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={chartData} margin={{ top: 10, right: 20, left: 0, bottom: 10 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#E2E8F0" />
          <XAxis
            dataKey="n_labeled"
            tick={{ fontSize: 12, fill: "#64748B" }}
            label={{
              value: "Annotation Budget (Labeled Samples)",
              position: "insideBottom",
              offset: -4,
              style: { fontSize: 11, fill: "#64748B" },
            }}
          />
          <YAxis
            tickFormatter={(v) => `${v}%`}
            tick={{ fontSize: 12, fill: "#64748B" }}
          />
          <Tooltip
            contentStyle={{
              backgroundColor: "#FFFFFF",
              borderColor: "#E2E8F0",
              borderRadius: "8px",
              fontSize: "12px",
            }}
            formatter={(val: number, name: string) => [`${val}%`, name]}
          />
          <Legend wrapperStyle={{ fontSize: "12px", paddingTop: "8px" }} />
          {strategies.map((strat) => (
            <Line
              key={strat}
              type="monotone"
              dataKey={strat}
              name={strat}
              stroke={STRATEGY_COLORS[strat] || "#2563EB"}
              strokeWidth={strat === "random" ? 2 : 2.5}
              strokeDasharray={strat === "random" ? "5 5" : undefined}
              dot={{ r: 3.5 }}
            />
          ))}
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}

