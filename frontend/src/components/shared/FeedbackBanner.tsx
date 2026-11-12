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
        "flex items-start gap-2.5 rounded-xl border border-[#FECACA] bg-[#FEF2F2] px-4 py-3 text-sm text-[#B91C1C]",
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
        "flex items-start gap-2.5 rounded-xl border border-[#FDE68A] bg-[#FFFBEB] px-4 py-3 text-xs text-[#92400E]",
        className
      )}
    >
      <Clock className="mt-0.5 h-4 w-4 shrink-0 text-[#B45309]" />
      <div>
        <span className="font-semibold">{title}: </span>
        <span>{description}</span>
      </div>
    </div>
  );
}

export function LoadingState({ label = "Loading data..." }: { label?: string }) {
  return (
    <div className="flex items-center justify-center gap-2.5 rounded-xl border border-[#E2E8F0] bg-white py-12 text-sm text-[#64748B]">
      <Loader2 className="h-4 w-4 animate-spin text-[#2563EB]" />
      <span>{label}</span>
    </div>
  );
}

