import * as React from "react";
import { cva, type VariantProps } from "class-variance-authority";
import { cn } from "@/lib/utils";

const buttonVariants = cva(
  "inline-flex items-center justify-center gap-2 whitespace-nowrap rounded-lg text-sm font-medium transition-all duration-150 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#6B3E26] focus-visible:ring-offset-1 disabled:pointer-events-none disabled:opacity-50",
  {
    variants: {
      variant: {
        default: "bg-[#6B3E26] text-[#FFFCF7] hover:bg-[#54301D] shadow-sm",
        teal: "bg-[#B98255] text-[#FFFCF7] hover:bg-[#9E6C42] shadow-sm",
        success: "bg-[#2D6A4F] text-[#FFFCF7] hover:bg-[#22533D] shadow-sm",
        outline:
          "border border-[#E6D6C6] bg-[#FFFCF7] text-[#38271F] hover:bg-[#FAF3EC] hover:border-[#B98255] hover:text-[#6B3E26] shadow-xs",
        ghost: "text-[#806D60] hover:bg-[#FAF3EC] hover:text-[#38271F]",
        destructive: "bg-[#B84034] text-white hover:bg-[#963127]",
      },
      size: {
        default: "h-9 px-4 py-2",
        sm: "h-8 rounded-lg px-3 text-xs",
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
