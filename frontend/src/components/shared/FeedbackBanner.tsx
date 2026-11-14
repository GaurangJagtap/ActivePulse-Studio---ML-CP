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
        "flex items-start gap-2.5 rounded-xl border border-[#D4C3F5] bg-[#EAE1FA]/80 px-4 py-3 text-xs text-[#4C1D95]",
        className
      )}
    >
      <Clock className="mt-0.5 h-4 w-4 shrink-0 text-[#6D28D9]" />
      <div>
        <span className="font-semibold">{title}: </span>
        <span>{description}</span>
      </div>
    </div>
  );
}

export function LoadingState({ label = "Loading data..." }: { label?: string }) {
  return (
    <div className="flex items-center justify-center gap-2.5 rounded-2xl border border-[#D5DDF2] bg-[#F3F5FD] py-12 text-sm text-[#64748B]">
      <Loader2 className="h-4 w-4 animate-spin text-[#4568D9]" />
      <span>{label}</span>
    </div>
  );
}

