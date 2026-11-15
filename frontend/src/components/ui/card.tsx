import * as React from "react";
import { cn } from "@/lib/utils";

export type CardTone =
  | "default"
  | "secondary"
  | "blue"
  | "lavender"
  | "mint";

const TONE_CLASSES: Record<CardTone, string> = {
  default: "border-[#E6D6C6] bg-[#FFFCF7]",
  secondary: "border-[#E6D6C6] bg-[#FFFCF7]",
  blue: "border-[#D8BEA6] bg-[#FAF3EC]",
  lavender: "border-[#E6D6C6] bg-[#FFFCF7]",
  mint: "border-[#C9D9C9] bg-[#F2F7F2]",
};

export interface CardProps extends React.HTMLAttributes<HTMLDivElement> {
  tone?: CardTone;
}

export const Card = React.forwardRef<HTMLDivElement, CardProps>(
  ({ className, tone = "default", ...props }, ref) => (
    <div
      ref={ref}
      className={cn(
        "rounded-xl border text-[#38271F] shadow-card transition-all duration-200 hover:border-[#D4BFAC] hover:shadow-card-hover",
        TONE_CLASSES[tone],
        className
      )}
      {...props}
    />
  )
);
Card.displayName = "Card";

export const CardHeader = React.forwardRef<
  HTMLDivElement,
  React.HTMLAttributes<HTMLDivElement>
>(({ className, ...props }, ref) => (
  <div
    ref={ref}
    className={cn("flex flex-col space-y-1 p-5 pb-3", className)}
    {...props}
  />
));
CardHeader.displayName = "CardHeader";

export const CardTitle = React.forwardRef<
  HTMLParagraphElement,
  React.HTMLAttributes<HTMLHeadingElement>
>(({ className, ...props }, ref) => (
  <h3
    ref={ref}
    className={cn(
      "text-sm font-semibold tracking-tight text-[#38271F]",
      className
    )}
    {...props}
  />
));
CardTitle.displayName = "CardTitle";

export const CardDescription = React.forwardRef<
  HTMLParagraphElement,
  React.HTMLAttributes<HTMLParagraphElement>
>(({ className, ...props }, ref) => (
  <p
    ref={ref}
    className={cn("text-xs text-[#806D60]", className)}
    {...props}
  />
));
CardDescription.displayName = "CardDescription";

export const CardContent = React.forwardRef<
  HTMLDivElement,
  React.HTMLAttributes<HTMLDivElement>
>(({ className, ...props }, ref) => (
  <div ref={ref} className={cn("p-5 pt-2", className)} {...props} />
));
CardContent.displayName = "CardContent";
