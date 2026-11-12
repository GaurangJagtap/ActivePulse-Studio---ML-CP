import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { RoundMetric } from "@/types/api";

interface LearningCurveChartProps {
  rounds: RoundMetric[];
  height?: number;
}

export function LearningCurveChart({
  rounds,
  height = 320,
}: LearningCurveChartProps) {
  if (!rounds.length) {
    return (
      <div className="flex h-[280px] items-center justify-center rounded-lg border border-dashed border-[#E2E8F0] bg-[#F8FAFC] text-xs text-[#64748B]">
        No round metrics recorded yet. Run an active learning experiment to plot the learning curve.
      </div>
    );
  }

  const data = rounds.map((r) => ({
    n_labeled: r.n_labeled,
    round: `R${r.round_num}`,
    accuracy: Number((r.test_acc * 100).toFixed(2)),
    f1: Number((r.test_f1 * 100).toFixed(2)),
    ece: r.ece !== null ? Number(r.ece.toFixed(4)) : null,
  }));

  return (
    <div style={{ width: "100%", height }}>
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={data} margin={{ top: 10, right: 20, left: 0, bottom: 10 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#E2E8F0" />
          <XAxis
            dataKey="n_labeled"
            tick={{ fontSize: 12, fill: "#64748B" }}
            label={{
              value: "Labeled Samples",
              position: "insideBottom",
              offset: -4,
              style: { fontSize: 11, fill: "#64748B" },
            }}
          />
          <YAxis
            domain={[0, 100]}
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
            formatter={(value: number, name: string) => [
              name === "ece" ? value : `${value}%`,
              name === "accuracy"
                ? "Holdout Accuracy"
                : name === "f1"
                  ? "Macro-F1"
                  : "ECE",
            ]}
            labelFormatter={(lbl) => `Labeled Samples: ${lbl}`}
          />
          <Legend wrapperStyle={{ fontSize: "12px", paddingTop: "8px" }} />
          <Line
            type="monotone"
            dataKey="accuracy"
            name="Holdout Accuracy"
            stroke="#2563EB"
            strokeWidth={2.5}
            dot={{ r: 4, fill: "#2563EB", strokeWidth: 1.5, stroke: "#FFFFFF" }}
            activeDot={{ r: 6 }}
          />
          <Line
            type="monotone"
            dataKey="f1"
            name="Macro-F1"
            stroke="#0D9488"
            strokeWidth={2}
            strokeDasharray="4 4"
            dot={{ r: 3.5, fill: "#0D9488" }}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}

