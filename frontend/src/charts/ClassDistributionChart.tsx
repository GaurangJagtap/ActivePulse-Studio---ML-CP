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
          <CartesianGrid strokeDasharray="3 3" stroke="#E2E8F0" vertical={false} />
          <XAxis
            dataKey="class_name"
            tick={{ fontSize: 11, fill: "#64748B" }}
            angle={-20}
            textAnchor="end"
            interval={0}
            height={50}
          />
          <YAxis tick={{ fontSize: 11, fill: "#64748B" }} />
          <Tooltip
            contentStyle={{
              backgroundColor: "#FFFFFF",
              borderColor: "#E2E8F0",
              borderRadius: "8px",
              fontSize: "12px",
            }}
          />
          <Bar dataKey="count" name="Samples" fill="#2563EB" radius={[4, 4, 0, 0]} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

