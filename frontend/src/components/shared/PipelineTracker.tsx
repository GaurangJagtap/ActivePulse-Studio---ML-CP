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
    <div className="rounded-xl border border-[#E6D6C6] bg-[#FFFCF7] p-4 shadow-card">
      <div className="grid grid-cols-1 gap-3 md:grid-cols-5">
        {stages.map((label, idx) => {
          const done = isCompleted || idx < activeIndex;
          const active = !isCompleted && idx === activeIndex;
          return (
            <div
              key={label}
              className={cn(
                "flex items-center gap-2.5 rounded-lg border px-3 py-2.5 transition-all",
                done
                  ? "border-[#B7D5C4] bg-[#EDF6F0]"
                  : active
                    ? "border-[#D8BEA6] bg-[#FAF3EC] shadow-xs"
                    : "border-[#E6D6C6] bg-[#F6EFE7]"
              )}
            >
              {done ? (
                <CheckCircle2 className="h-4 w-4 shrink-0 text-[#2D6A4F]" />
              ) : active && isRunning ? (
                <Loader2 className="h-4 w-4 shrink-0 animate-spin text-[#6B3E26]" />
              ) : active ? (
                <span className="flex h-4 w-4 shrink-0 items-center justify-center rounded-full bg-[#6B3E26] font-mono text-[10px] font-semibold text-white">
                  {idx + 1}
                </span>
              ) : (
                <Circle className="h-4 w-4 shrink-0 text-[#A89A8F]" />
              )}
              <div className="min-w-0">
                <p className="font-mono text-[10px] uppercase tracking-wider text-[#806D60]">
                  Step {idx + 1}
                </p>
                <p
                  className={cn(
                    "truncate text-xs font-medium",
                    done
                      ? "text-[#2D6A4F]"
                      : active
                        ? "text-[#38271F]"
                        : "text-[#806D60]"
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

