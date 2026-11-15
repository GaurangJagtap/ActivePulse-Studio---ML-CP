import { AlertCircle, Clock, Loader2 } from "lucide-react";
import { cn } from "@/lib/utils";

export function ErrorBanner({
  message,
  className,
}: {
  message: string;
  className?: string;
}) {
  return (
    <div
      className={cn(
        "flex items-start gap-2.5 rounded-xl border border-[#F1BDB7] bg-[#FDF0EE] px-4 py-3 text-sm text-[#A63529] shadow-xs",
        className
      )}
    >
      <AlertCircle className="mt-0.5 h-4 w-4 shrink-0" />
      <span>{message}</span>
    </div>
  );
}

export function PendingFeatureBanner({
  title,
  description,
  className,
}: {
  title: string;
  description: string;
  className?: string;
}) {
  return (
    <div
      className={cn(
        "flex items-start gap-2.5 rounded-xl border border-[#D8BEA6] bg-[#FAF3EC] px-4 py-3 text-xs text-[#6B3E26] shadow-xs",
        className
      )}
    >
      <Clock className="mt-0.5 h-4 w-4 shrink-0 text-[#B98255]" />
      <div>
        <span className="font-semibold">{title}: </span>
        <span>{description}</span>
      </div>
    </div>
  );
}

export function LoadingState({ label = "Loading data..." }: { label?: string }) {
  return (
    <div className="flex items-center justify-center gap-2.5 rounded-2xl border border-[#E6D6C6] bg-[#FFFCF7] py-12 text-sm text-[#806D60] shadow-card">
      <Loader2 className="h-4 w-4 animate-spin text-[#6B3E26]" />
      <span>{label}</span>
    </div>
  );
}

