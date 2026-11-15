import {
  Bar,
  BarChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { ClassCount } from "@/types/api";

export function ClassDistributionChart({
  data,
  height = 280,
}: {
  data: ClassCount[];
  height?: number;
}) {
  return (
    <div style={{ width: "100%", height }}>
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={data} margin={{ top: 10, right: 16, left: 0, bottom: 24 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#EFE4D8" vertical={false} />
          <XAxis
            dataKey="class_name"
            tick={{ fontSize: 11, fill: "#806D60" }}
            axisLine={{ stroke: "#E6D6C6" }}
            tickLine={{ stroke: "#E6D6C6" }}
            angle={-20}
            textAnchor="end"
            interval={0}
            height={50}
          />
          <YAxis
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
          />
          <Bar dataKey="count" name="Samples" fill="#6B3E26" radius={[4, 4, 0, 0]} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

