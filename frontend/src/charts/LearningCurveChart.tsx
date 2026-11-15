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
      <div className="flex h-[300px] items-center justify-center rounded-xl border border-dashed border-[#E6D6C6] bg-[#FFFCF7] text-xs text-[#806D60]">
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
          <CartesianGrid strokeDasharray="3 3" stroke="#EFE4D8" />
          <XAxis
            dataKey="n_labeled"
            tick={{ fontSize: 12, fill: "#806D60" }}
            axisLine={{ stroke: "#E6D6C6" }}
            tickLine={{ stroke: "#E6D6C6" }}
            label={{
              value: "Labeled Samples",
              position: "insideBottom",
              offset: -6,
              style: { fontSize: 11, fill: "#806D60", fontWeight: 500 },
            }}
          />
          <YAxis
            domain={[0, 100]}
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
          <Legend wrapperStyle={{ fontSize: "12px", paddingTop: "10px", color: "#38271F" }} />
          <Line
            type="monotone"
            dataKey="accuracy"
            name="Holdout Accuracy"
            stroke="#6B3E26"
            strokeWidth={2.75}
            dot={{ r: 4, fill: "#6B3E26", strokeWidth: 1.5, stroke: "#FFFCF7" }}
            activeDot={{ r: 6 }}
          />
          <Line
            type="monotone"
            dataKey="f1"
            name="Macro-F1"
            stroke="#B98255"
            strokeWidth={2}
            strokeDasharray="4 4"
            dot={{ r: 3.5, fill: "#B98255" }}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}

