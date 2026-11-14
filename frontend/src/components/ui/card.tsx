import * as React from "react";
import { cn } from "@/lib/utils";

export type CardTone =
  | "default"
  | "secondary"
  | "blue"
  | "lavender"
  | "mint";

const TONE_CLASSES: Record<CardTone, string> = {
  default: "border-[#D5DDF2] bg-[#F3F5FD]",
  secondary: "border-[#D5DDF2] bg-[#EEF1FB]",
  blue: "border-[#C7D5FA] bg-[#DCE5FF]/65",
  lavender: "border-[#D9CDF6] bg-[#EAE1FA]/65",
  mint: "border-[#BDE5DA] bg-[#D9F0EA]/65",
};

export interface CardProps extends React.HTMLAttributes<HTMLDivElement> {
  tone?: CardTone;
}

export const Card = React.forwardRef<HTMLDivElement, CardProps>(
  ({ className, tone = "default", ...props }, ref) => (
    <div
      ref={ref}
      className={cn(
        "rounded-2xl border text-[#263452] shadow-card transition-all duration-200",
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
      "text-sm font-semibold tracking-tight text-[#263452]",
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
    className={cn("text-xs text-[#64748B]", className)}
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
