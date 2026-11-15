import * as React from "react";
import { cn } from "@/lib/utils";

export const Input = React.forwardRef<
  HTMLInputElement,
  React.InputHTMLAttributes<HTMLInputElement>
>(({ className, ...props }, ref) => (
  <input
    ref={ref}
    className={cn(
      "flex h-9 w-full rounded-lg border border-[#E6D6C6] bg-[#FFFCF7] px-3 py-1.5 text-sm text-[#38271F] placeholder:text-[#806D60]/70 focus:border-[#6B3E26] focus:bg-[#FFFFFF] focus:outline-none focus:ring-2 focus:ring-[#6B3E26]/15 disabled:cursor-not-allowed disabled:opacity-50",
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
      "flex h-9 w-full rounded-lg border border-[#E6D6C6] bg-[#FFFCF7] px-3 py-1.5 text-sm text-[#38271F] focus:border-[#6B3E26] focus:bg-[#FFFFFF] focus:outline-none focus:ring-2 focus:ring-[#6B3E26]/15 disabled:cursor-not-allowed disabled:opacity-50",
      className
    )}
    {...props}
  >
    {children}
  </select>
));
Select.displayName = "Select";
