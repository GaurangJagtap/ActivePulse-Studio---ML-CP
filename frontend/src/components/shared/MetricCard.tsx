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
  blue: "bg-[#DCE5FF] text-[#4568D9]",
  teal: "bg-[#D9F0EA] text-[#159C91]",
  green: "bg-[#D9F0EA] text-[#0F766E]",
  lavender: "bg-[#EAE1FA] text-[#5B21B6]",
  slate: "bg-[#E3E9FA] text-[#263452]",
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
    <Card tone={tone} className="transition-colors hover:border-[#BFCBEF]">
      <CardContent className="p-5">
        <div className="flex items-start justify-between gap-3">
          <div className="min-w-0 space-y-1.5">
            <p className="text-[11px] font-semibold uppercase tracking-wider text-[#64748B]">
              {title}
            </p>
            <div className="flex items-baseline gap-2">
              <span className="font-mono text-2xl font-semibold tracking-tight text-[#263452]">
                {value}
              </span>
              {delta && (
                <span className="font-mono text-xs font-medium text-[#159C91]">
                  {delta}
                </span>
              )}
            </div>
            {subtitle && (
              <p className="truncate text-xs text-[#64748B]" title={subtitle}>
                {subtitle}
              </p>
            )}
          </div>
          <div
            className={cn(
              "flex h-10 w-10 shrink-0 items-center justify-center rounded-xl",
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

