import * as React from "react";
import { cva, type VariantProps } from "class-variance-authority";
import { cn } from "@/lib/utils";

const buttonVariants = cva(
  "inline-flex items-center justify-center gap-2 whitespace-nowrap rounded-lg text-sm font-medium transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#2563EB] focus-visible:ring-offset-1 disabled:pointer-events-none disabled:opacity-50",
  {
    variants: {
      variant: {
        default: "bg-[#2563EB] text-white hover:bg-[#1D4ED8] shadow-sm",
        teal: "bg-[#0D9488] text-white hover:bg-[#0F766E] shadow-sm",
        success: "bg-[#16A34A] text-white hover:bg-[#15803D] shadow-sm",
        outline:
          "border border-[#E2E8F0] bg-white text-[#172033] hover:bg-[#F8FAFC]",
        ghost: "text-[#64748B] hover:bg-[#F1F5F9] hover:text-[#172033]",
        destructive: "bg-[#DC2626] text-white hover:bg-[#B91C1C]",
      },
      size: {
        default: "h-9 px-4 py-2",
        sm: "h-8 rounded-md px-3 text-xs",
        lg: "h-10 rounded-lg px-5 text-sm",
        icon: "h-9 w-9",
      },
    },
    defaultVariants: {
      variant: "default",
      size: "default",
    },
  }
);

export interface ButtonProps
  extends React.ButtonHTMLAttributes<HTMLButtonElement>,
    VariantProps<typeof buttonVariants> {}

export const Button = React.forwardRef<HTMLButtonElement, ButtonProps>(
  ({ className, variant, size, ...props }, ref) => (
    <button
      ref={ref}
      className={cn(buttonVariants({ variant, size, className }))}
      {...props}
    />
  )
);
Button.displayName = "Button";

