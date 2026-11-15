import * as React from "react";
import { cva, type VariantProps } from "class-variance-authority";
import { cn } from "@/lib/utils";

const badgeVariants = cva(
  "inline-flex items-center gap-1.5 rounded-full border px-2.5 py-0.5 text-xs font-medium transition-colors",
  {
    variants: {
      variant: {
        default: "border-[#D8BEA6] bg-[#FAF3EC] text-[#6B3E26]",
        lavender: "border-[#E6D6C6] bg-[#F1E8DF] text-[#6B3E26]",
        teal: "border-[#DFC6AF] bg-[#FAF1E8] text-[#8C5832]",
        success: "border-[#B7D5C4] bg-[#EDF6F0] text-[#2D6A4F]",
        warning: "border-[#EAD3A6] bg-[#FDF8ED] text-[#9A6218]",
        danger: "border-[#F1BDB7] bg-[#FDF0EE] text-[#A63529]",
        neutral: "border-[#E6D6C6] bg-[#F6EFE7] text-[#806D60]",
      },
    },
    defaultVariants: {
      variant: "neutral",
    },
  }
);

export interface BadgeProps
  extends React.HTMLAttributes<HTMLSpanElement>,
    VariantProps<typeof badgeVariants> {}

export function Badge({ className, variant, ...props }: BadgeProps) {
  return <span className={cn(badgeVariants({ variant, className }))} {...props} />;
}
