import type { LucideIcon } from "lucide-react";
import { Card, CardContent } from "@/components/ui/card";
import { cn } from "@/lib/utils";

interface MetricCardProps {
  title: string;
  value: string;
  subtitle?: string;
  icon: LucideIcon;
  accent?: "blue" | "teal" | "green" | "slate";
  delta?: string;
}

const ACCENT_MAP = {
  blue: "bg-[#EFF6FF] text-[#2563EB]",
  teal: "bg-[#F0FDFA] text-[#0D9488]",
  green: "bg-[#F0FDF4] text-[#16A34A]",
  slate: "bg-[#F1F5F9] text-[#475569]",
};

export function MetricCard({
  title,
  value,
  subtitle,
  icon: Icon,
  accent = "blue",
  delta,
}: MetricCardProps) {
  return (
    <Card>
      <CardContent className="p-5">
        <div className="flex items-start justify-between gap-3">
          <div className="space-y-1.5">
            <p className="text-xs font-medium uppercase tracking-wider text-[#64748B]">
              {title}
            </p>
            <div className="flex items-baseline gap-2">
              <span className="font-mono text-2xl font-semibold tracking-tight text-[#172033]">
                {value}
              </span>
              {delta && (
                <span className="font-mono text-xs font-medium text-[#16A34A]">
                  {delta}
                </span>
              )}
            </div>
            {subtitle && (
              <p className="text-xs text-[#64748B]">{subtitle}</p>
            )}
          </div>
          <div
            className={cn(
              "flex h-10 w-10 shrink-0 items-center justify-center rounded-lg",
              ACCENT_MAP[accent]
            )}
          >
            <Icon className="h-5 w-5" />
          </div>
        </div>
      </CardContent>
    </Card>
  );
}

