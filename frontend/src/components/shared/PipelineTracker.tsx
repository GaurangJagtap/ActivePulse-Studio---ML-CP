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
    <div className="rounded-2xl border border-[#D5DDF2] bg-[#F3F5FD] p-4 shadow-card">
      <div className="grid grid-cols-1 gap-3 md:grid-cols-5">
        {stages.map((label, idx) => {
          const done = isCompleted || idx < activeIndex;
          const active = !isCompleted && idx === activeIndex;
          return (
            <div
              key={label}
              className={cn(
                "flex items-center gap-2.5 rounded-xl border px-3 py-2.5 transition-all",
                done
                  ? "border-[#9CDFD0] bg-[#D9F0EA]/75"
                  : active
                    ? "border-[#AEBEFB] bg-[#DCE5FF]/85 shadow-sm"
                    : "border-[#D5DDF2] bg-[#EEF1FB]/70"
              )}
            >
              {done ? (
                <CheckCircle2 className="h-4 w-4 shrink-0 text-[#159C91]" />
              ) : active && isRunning ? (
                <Loader2 className="h-4 w-4 shrink-0 animate-spin text-[#4568D9]" />
              ) : active ? (
                <span className="flex h-4 w-4 shrink-0 items-center justify-center rounded-full bg-[#4568D9] font-mono text-[10px] font-semibold text-white">
                  {idx + 1}
                </span>
              ) : (
                <Circle className="h-4 w-4 shrink-0 text-[#8B9BB4]" />
              )}
              <div className="min-w-0">
                <p className="font-mono text-[10px] uppercase tracking-wider text-[#64748B]">
                  Step {idx + 1}
                </p>
                <p
                  className={cn(
                    "truncate text-xs font-medium",
                    done
                      ? "text-[#0F766E]"
                      : active
                        ? "text-[#263452]"
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

