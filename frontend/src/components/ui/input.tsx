import * as React from "react";
import { cn } from "@/lib/utils";

export const Input = React.forwardRef<
  HTMLInputElement,
  React.InputHTMLAttributes<HTMLInputElement>
>(({ className, ...props }, ref) => (
  <input
    ref={ref}
    className={cn(
      "flex h-9 w-full rounded-lg border border-[#E2E8F0] bg-white px-3 py-1.5 text-sm text-[#172033] placeholder:text-[#94A3B8] focus:border-[#2563EB] focus:outline-none focus:ring-2 focus:ring-[#2563EB]/15 disabled:cursor-not-allowed disabled:opacity-50",
      className
    )}
    {...props}
  />
));
Input.displayName = "Input";

export const Select = React.forwardRef<
  HTMLSelectElement,
  React.SelectHTMLAttributes<HTMLSelectElement>
>(({ className, children, ...props }, ref) => (
  <select
    ref={ref}
    className={cn(
      "flex h-9 w-full rounded-lg border border-[#E2E8F0] bg-white px-3 py-1.5 text-sm text-[#172033] focus:border-[#2563EB] focus:outline-none focus:ring-2 focus:ring-[#2563EB]/15 disabled:cursor-not-allowed disabled:opacity-50",
      className
    )}
    {...props}
  >
    {children}
  </select>
));
Select.displayName = "Select";

