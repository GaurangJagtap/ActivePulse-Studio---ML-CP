import * as React from "react";
import { cva, type VariantProps } from "class-variance-authority";
import { cn } from "@/lib/utils";

const buttonVariants = cva(
  "inline-flex items-center justify-center gap-2 whitespace-nowrap rounded-xl text-sm font-medium transition-all duration-150 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#4568D9] focus-visible:ring-offset-1 disabled:pointer-events-none disabled:opacity-50",
  {
    variants: {
      variant: {
        default: "bg-[#4568D9] text-white hover:bg-[#3859C7] shadow-sm",
        teal: "bg-[#159C91] text-white hover:bg-[#118279] shadow-sm",
        success: "bg-[#159C91] text-white hover:bg-[#118279] shadow-sm",
        outline:
          "border border-[#CBD5EE] bg-[#F3F5FD] text-[#263452] hover:bg-[#DCE5FF]/60 hover:border-[#718BE0]",
        ghost: "text-[#64748B] hover:bg-[#DCE5FF]/50 hover:text-[#263452]",
        destructive: "bg-[#DC2626] text-white hover:bg-[#B91C1C]",
      },
      size: {
        default: "h-9 px-4 py-2",
        sm: "h-8 rounded-lg px-3 text-xs",
        lg: "h-10 rounded-xl px-5 text-sm",
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
