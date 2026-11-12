import { CheckCircle2, Circle, Loader2 } from "lucide-react";
import { cn } from "@/lib/utils";

const DEFAULT_STAGES = [
  "Unlabeled Data Intake",
  "PyTorch Retraining",
  "Temperature Calibration",
  "Uncertainty & Diversity Scoring",
  "Holdout Evaluation (Zero Leakage)",
];

interface PipelineTrackerProps {
  stages?: string[];
  activeIndex: number;
  isRunning: boolean;
  isCompleted: boolean;
}

export function PipelineTracker({
  stages = DEFAULT_STAGES,
  activeIndex,
  isRunning,
  isCompleted,
}: PipelineTrackerProps) {
  return (
    <div className="rounded-xl border border-[#E2E8F0] bg-white p-4 shadow-card">
      <div className="grid grid-cols-1 gap-3 md:grid-cols-5">
        {stages.map((label, idx) => {
          const done = isCompleted || idx < activeIndex;
          const active = !isCompleted && idx === activeIndex;
          return (
            <div
              key={label}
              className={cn(
                "flex items-center gap-2.5 rounded-lg border px-3 py-2.5 transition-colors",
                done
                  ? "border-[#BBF7D0] bg-[#F0FDF4]"
                  : active
                    ? "border-[#BFDBFE] bg-[#EFF6FF]"
                    : "border-[#E2E8F0] bg-[#F8FAFC]"
              )}
            >
              {done ? (
                <CheckCircle2 className="h-4 w-4 shrink-0 text-[#16A34A]" />
              ) : active && isRunning ? (
                <Loader2 className="h-4 w-4 shrink-0 animate-spin text-[#2563EB]" />
              ) : active ? (
                <span className="flex h-4 w-4 shrink-0 items-center justify-center rounded-full bg-[#2563EB] font-mono text-[10px] font-semibold text-white">
                  {idx + 1}
                </span>
              ) : (
                <Circle className="h-4 w-4 shrink-0 text-[#94A3B8]" />
              )}
              <div className="min-w-0">
                <p className="font-mono text-[10px] uppercase tracking-wider text-[#64748B]">
                  Step {idx + 1}
                </p>
                <p
                  className={cn(
                    "truncate text-xs font-medium",
                    done
                      ? "text-[#15803D]"
                      : active
                        ? "text-[#1D4ED8]"
                        : "text-[#64748B]"
                  )}
                  title={label}
                >
                  {label}
                </p>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

