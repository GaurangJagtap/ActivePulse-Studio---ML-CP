import * as React from "react";
import { cva, type VariantProps } from "class-variance-authority";
import { cn } from "@/lib/utils";

const badgeVariants = cva(
  "inline-flex items-center gap-1.5 rounded-full border px-2.5 py-0.5 text-xs font-medium transition-colors",
  {
    variants: {
      variant: {
        default: "border-[#B8C9F8] bg-[#DCE5FF] text-[#2F4EB8]",
        lavender: "border-[#D2C1F5] bg-[#EAE1FA] text-[#55389B]",
        teal: "border-[#A9DFD4] bg-[#D9F0EA] text-[#0E746C]",
        success: "border-[#A9DFD4] bg-[#D9F0EA] text-[#0E746C]",
        warning: "border-[#F3D8A1] bg-[#FDF3DB] text-[#9A5B13]",
        danger: "border-[#F5BDBD] bg-[#FDECEC] text-[#B42318]",
        neutral: "border-[#D5DDF2] bg-[#EEF1FB] text-[#475569]",
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
