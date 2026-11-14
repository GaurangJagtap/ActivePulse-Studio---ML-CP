import * as React from "react";
import { cn } from "@/lib/utils";

export const Input = React.forwardRef<
  HTMLInputElement,
  React.InputHTMLAttributes<HTMLInputElement>
>(({ className, ...props }, ref) => (
  <input
    ref={ref}
    className={cn(
      "flex h-9 w-full rounded-xl border border-[#CBD5EE] bg-[#F8FAFF] px-3 py-1.5 text-sm text-[#263452] placeholder:text-[#64748B]/70 focus:border-[#4568D9] focus:bg-[#FFFFFF] focus:outline-none focus:ring-2 focus:ring-[#4568D9]/20 disabled:cursor-not-allowed disabled:opacity-50",
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
      "flex h-9 w-full rounded-xl border border-[#CBD5EE] bg-[#F8FAFF] px-3 py-1.5 text-sm text-[#263452] focus:border-[#4568D9] focus:bg-[#FFFFFF] focus:outline-none focus:ring-2 focus:ring-[#4568D9]/20 disabled:cursor-not-allowed disabled:opacity-50",
      className
    )}
    {...props}
  >
    {children}
  </select>
));
Select.displayName = "Select";
