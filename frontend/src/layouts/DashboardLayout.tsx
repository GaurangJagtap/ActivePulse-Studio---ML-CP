import React from "react";
import {
  BarChart3,
  Database,
  FlaskConical,
  History,
  LayoutDashboard,
  PenTool,
  RefreshCw,
  Sparkles,
  Layers,
  ChevronDown,
} from "lucide-react";
import { cn } from "@/lib/utils";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import type {
  ExperimentSummary,
  HealthResponse,
  NavSectionId,
} from "@/types/api";

interface NavItem {
  id: NavSectionId;
  label: string;
  shortLabel: string;
  badge?: string;
  icon: typeof LayoutDashboard;
}

export const NAV_ITEMS: NavItem[] = [
  {
    id: "overview",
    label: "Overview",
    shortLabel: "Overview",
    icon: LayoutDashboard,
  },
  {
    id: "datasets",
    label: "Datasets",
    shortLabel: "Data",
    icon: Database,
  },
  {
    id: "lab",
    label: "Active Learning Lab",
    shortLabel: "AL Lab",
    badge: "Core",
    icon: FlaskConical,
  },
  {
    id: "annotation",
    label: "Annotation Studio",
    shortLabel: "Labeling",
    badge: "HITL",
    icon: PenTool,
  },
  {
    id: "benchmarks",
    label: "Benchmark Arena",
    shortLabel: "Arena",
    badge: "8 Models",
    icon: BarChart3,
  },
  {
    id: "history",
    label: "Experiment Ledger",
    shortLabel: "History",
    icon: History,
  },
];

interface DashboardLayoutProps {
  activeSection: NavSectionId;
  onSelectSection: (section: NavSectionId) => void;
  experiments: ExperimentSummary[];
  selectedExperimentId: string | null;
  onSelectExperiment: (id: string) => void;
  health: HealthResponse | null;
  onRefreshAll: () => void;
  children: React.ReactNode;
}

export function DashboardLayout({
  activeSection,
  onSelectSection,
  experiments,
  selectedExperimentId,
  onSelectExperiment,
  health,
  onRefreshAll,
  children,
}: DashboardLayoutProps) {
  const currentNav =
    NAV_ITEMS.find((n) => n.id === activeSection) || NAV_ITEMS[0];

  return (
    <div className="aurora-canvas min-h-screen text-[#263452]">
      {/* Modern Floating Header Bar */}
      <header className="sticky top-0 z-40 w-full border-b border-[#D5DDF2]/80 bg-[#EEF1FB]/85 backdrop-blur-md shadow-xs">
        <div className="mx-auto flex h-16 max-w-[1536px] items-center justify-between px-4 sm:px-6">
          {/* Brand Logo & Title */}
          <div className="flex items-center gap-3">
            <div className="relative flex h-10 w-10 shrink-0 items-center justify-center rounded-2xl bg-gradient-to-tr from-[#3b59c7] via-[#4568D9] to-[#159C91] text-white shadow-md shadow-[#4568D9]/20 ring-2 ring-white/50">
              <svg
                width="20"
                height="20"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="2.5"
                strokeLinecap="round"
                strokeLinejoin="round"
              >
                <polyline points="22 12 18 12 15 20 9 4 6 12 2 12" />
              </svg>
              <span className="absolute -top-1 -right-1 flex h-2.5 w-2.5">
                <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-teal-400 opacity-75"></span>
                <span className="relative inline-flex h-2.5 w-2.5 rounded-full bg-teal-500"></span>
              </span>
            </div>

            <div>
              <div className="flex items-center gap-2">
                <span className="text-base font-bold tracking-tight text-[#17243A]">
                  ActivePulse
                </span>
                <span className="rounded-md bg-gradient-to-r from-[#4568D9]/15 to-[#159C91]/15 px-1.5 py-0.5 text-[10px] font-bold text-[#4568D9]">
                  STUDIO
                </span>
              </div>
              <p className="hidden text-[11px] font-medium text-[#64748B] sm:block">
                AI Active Learning & Annotation Cost Reduction
              </p>
            </div>
          </div>

          {/* Modern Top Navigation Pill Bar */}
          <nav className="hidden lg:flex items-center rounded-2xl border border-[#D5DDF2] bg-white/70 p-1 shadow-xs backdrop-blur-md">
            {NAV_ITEMS.map((item) => {
              const Icon = item.icon;
              const isActive = item.id === activeSection;
              return (
                <button
                  key={item.id}
                  type="button"
                  onClick={() => onSelectSection(item.id)}
                  className={cn(
                    "group relative flex items-center gap-2 rounded-xl px-3.5 py-1.5 text-xs font-semibold transition-all duration-200",
                    isActive
                      ? "bg-[#17243A] text-white shadow-sm"
                      : "text-[#475569] hover:bg-[#EEF1FB] hover:text-[#17243A]"
                  )}
                >
                  <Icon
                    className={cn(
                      "h-3.5 w-3.5 transition-transform group-hover:scale-110",
                      isActive ? "text-[#5EEAD4]" : "text-[#718BE0]"
                    )}
                  />
                  <span>{item.label}</span>
                  {item.badge && (
                    <span
                      className={cn(
                        "rounded px-1.5 py-0.2 text-[9px] font-bold tracking-wide",
                        isActive
                          ? "bg-[#5EEAD4]/20 text-[#5EEAD4]"
                          : "bg-[#DCE5FF] text-[#4568D9]"
                      )}
                    >
                      {item.badge}
                    </span>
                  )}
                </button>
              );
            })}
          </nav>

          {/* Quick Controls & Status Pill */}
          <div className="flex items-center gap-2.5">
            {/* Active Run Selector */}
            {experiments.length > 0 && (
              <div className="hidden xl:flex items-center gap-1.5 rounded-xl border border-[#D5DDF2] bg-white/70 px-2.5 py-1 shadow-2xs">
                <Layers className="h-3.5 w-3.5 text-[#4568D9]" />
                <span className="text-[11px] font-semibold text-[#64748B]">
                  Run:
                </span>
                <select
                  value={selectedExperimentId || ""}
                  onChange={(e) => onSelectExperiment(e.target.value)}
                  className="bg-transparent font-medium text-xs text-[#17243A] outline-none cursor-pointer max-w-[140px] truncate"
                >
                  {experiments.map((exp) => (
                    <option key={exp.id} value={exp.id}>
                      {exp.name} ({exp.strategy})
                    </option>
                  ))}
                </select>
              </div>
            )}

            {/* System Status Pill */}
            <div
              className={cn(
                "hidden sm:flex items-center gap-1.5 rounded-xl border px-2.5 py-1 text-[11px] font-medium shadow-2xs",
                health
                  ? "border-teal-200 bg-teal-50/80 text-teal-800"
                  : "border-amber-200 bg-amber-50/80 text-amber-800"
              )}
              title={
                health
                  ? `PyTorch ${health.pytorch_version} (${health.device.toUpperCase()}) · ${health.registered_strategies} strategies`
                  : "Backend connecting..."
              }
            >
              <span
                className={cn(
                  "h-2 w-2 rounded-full",
                  health ? "bg-teal-500 animate-pulse" : "bg-amber-500"
                )}
              />
              <span className="font-semibold">
                {health ? "PyTorch Ready" : "Connecting..."}
              </span>
            </div>

            {/* Refresh Button */}
            <Button
              variant="outline"
              size="sm"
              onClick={onRefreshAll}
              className="h-8 rounded-xl border-[#D5DDF2] bg-white/80 hover:bg-white text-xs font-semibold shadow-2xs hover:shadow-xs transition-all"
              title="Sync metrics and status from backend"
            >
              <RefreshCw className="h-3.5 w-3.5 text-[#4568D9]" />
              <span className="hidden md:inline">Sync</span>
            </Button>
          </div>
        </div>

        {/* Mobile/Tablet Horizontal Scroll Nav Tabs */}
        <div className="flex lg:hidden overflow-x-auto border-t border-[#D5DDF2]/70 bg-white/50 px-3 py-1.5 no-scrollbar gap-1">
          {NAV_ITEMS.map((item) => {
            const Icon = item.icon;
            const isActive = item.id === activeSection;
            return (
              <button
                key={item.id}
                type="button"
                onClick={() => onSelectSection(item.id)}
                className={cn(
                  "flex shrink-0 items-center gap-1.5 rounded-lg px-2.5 py-1 text-xs font-medium transition-all",
                  isActive
                    ? "bg-[#17243A] text-white shadow-xs font-semibold"
                    : "text-[#64748B] hover:bg-[#EEF1FB] hover:text-[#17243A]"
                )}
              >
                <Icon className={cn("h-3.5 w-3.5", isActive ? "text-[#5EEAD4]" : "text-[#718BE0]")} />
                <span>{item.shortLabel}</span>
              </button>
            );
          })}
        </div>
      </header>

      {/* Main Content Area */}
      <main className="mx-auto w-full max-w-[1536px] p-4 sm:p-6 lg:p-8">
        {children}
      </main>
    </div>
  );
}
