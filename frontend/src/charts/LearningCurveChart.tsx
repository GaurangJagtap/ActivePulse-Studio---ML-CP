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
      <div className="flex h-[300px] items-center justify-center rounded-xl border border-dashed border-[#C7D3F0] bg-[#EEF1FB]/80 text-xs text-[#64748B]">
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
        <LineChart data={data} margin={{ top: 12, right: 24, left: 4, bottom: 14 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#D5DDF2" />
          <XAxis
            dataKey="n_labeled"
            tick={{ fontSize: 12, fill: "#64748B" }}
            axisLine={{ stroke: "#C7D3F0" }}
            tickLine={{ stroke: "#C7D3F0" }}
            label={{
              value: "Labeled Samples",
              position: "insideBottom",
              offset: -6,
              style: { fontSize: 11, fill: "#64748B", fontWeight: 500 },
            }}
          />
          <YAxis
            domain={[0, 100]}
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
          <Legend wrapperStyle={{ fontSize: "12px", paddingTop: "10px", color: "#263452" }} />
          <Line
            type="monotone"
            dataKey="accuracy"
            name="Holdout Accuracy"
            stroke="#4568D9"
            strokeWidth={2.75}
            dot={{ r: 4, fill: "#4568D9", strokeWidth: 1.5, stroke: "#F3F5FD" }}
            activeDot={{ r: 6 }}
          />
          <Line
            type="monotone"
            dataKey="f1"
            name="Macro-F1"
            stroke="#159C91"
            strokeWidth={2.25}
            strokeDasharray="4 4"
            dot={{ r: 3.5, fill: "#159C91" }}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}

