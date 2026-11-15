import type { LucideIcon } from "lucide-react";
import { Card, CardContent, type CardTone } from "@/components/ui/card";
import { cn } from "@/lib/utils";

interface MetricCardProps {
  title: string;
  value: string;
  subtitle?: string;
  icon: LucideIcon;
  accent?: "blue" | "teal" | "green" | "lavender" | "slate";
  tone?: CardTone;
  delta?: string;
}

const ACCENT_MAP = {
  blue: "bg-[#FAF3EC] text-[#6B3E26]",
  teal: "bg-[#FAF1E8] text-[#B98255]",
  green: "bg-[#EDF6F0] text-[#2D6A4F]",
  lavender: "bg-[#F3EBE3] text-[#6B3E26]",
  slate: "bg-[#EFE6DC] text-[#38271F]",
};

export function MetricCard({
  title,
  value,
  subtitle,
  icon: Icon,
  accent = "blue",
  tone = "default",
  delta,
}: MetricCardProps) {
  return (
    <Card tone={tone} className="transition-all hover:border-[#D4BFAC] hover:shadow-card-hover">
      <CardContent className="p-5">
        <div className="flex items-start justify-between gap-3">
          <div className="min-w-0 space-y-1.5">
            <p className="text-[11px] font-semibold uppercase tracking-wider text-[#806D60]">
              {title}
            </p>
            <div className="flex items-baseline gap-2">
              <span className="font-mono text-2xl font-semibold tracking-tight text-[#38271F]">
                {value}
              </span>
              {delta && (
                <span className="font-mono text-xs font-medium text-[#B98255]">
                  {delta}
                </span>
              )}
            </div>
            {subtitle && (
              <p className="truncate text-xs text-[#806D60]" title={subtitle}>
                {subtitle}
              </p>
            )}
          </div>
          <div
            className={cn(
              "flex h-10 w-10 shrink-0 items-center justify-center rounded-xl border border-[#E6D6C6]",
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

