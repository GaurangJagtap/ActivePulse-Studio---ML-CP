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
          <CartesianGrid strokeDasharray="3 3" stroke="#D5DDF2" vertical={false} />
          <XAxis
            dataKey="class_name"
            tick={{ fontSize: 11, fill: "#64748B" }}
            axisLine={{ stroke: "#C7D3F0" }}
            tickLine={{ stroke: "#C7D3F0" }}
            angle={-20}
            textAnchor="end"
            interval={0}
            height={50}
          />
          <YAxis
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
          />
          <Bar dataKey="count" name="Samples" fill="#4568D9" radius={[6, 6, 0, 0]} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

