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
    icon: FlaskConical,
  },
  {
    id: "annotation",
    label: "Annotation Studio",
    shortLabel: "Labeling",
    icon: PenTool,
  },
  {
    id: "benchmarks",
    label: "Benchmark Arena",
    shortLabel: "Arena",
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
    <div className="aurora-canvas min-h-screen text-[#38271F] bg-[#F6EFE7]">
      {/* Modern Floating Header Bar */}
      <header className="sticky top-0 z-40 w-full border-b border-[#E6D6C6] bg-[#FFFCF7]/95 backdrop-blur-md shadow-header">
        <div className="mx-auto flex h-16 max-w-[1536px] items-center justify-between px-4 sm:px-6">
          {/* Brand Logo & Title */}
          <div className="flex items-center gap-3">
            <div className="relative flex h-10 w-10 shrink-0 items-center justify-center rounded-xl bg-gradient-to-tr from-[#54301D] via-[#6B3E26] to-[#B98255] text-[#FFFCF7] shadow-sm">
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
                <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-[#B98255] opacity-75"></span>
                <span className="relative inline-flex h-2.5 w-2.5 rounded-full bg-[#6B3E26]"></span>
              </span>
            </div>

            <div>
              <div className="flex items-center gap-2">
                <span className="text-base font-bold tracking-tight text-[#38271F]">
                  ActivePulse
                </span>
                <span className="rounded-md bg-[#FAF3EC] px-1.5 py-0.5 text-[10px] font-bold text-[#6B3E26] border border-[#D8BEA6]">
                  STUDIO
                </span>
              </div>
              <p className="hidden text-[11px] font-medium text-[#806D60] sm:block">
                AI Active Learning & Annotation Cost Reduction
              </p>
            </div>
          </div>

          {/* Modern Top Navigation Pill Bar */}
          <nav className="hidden lg:flex items-center gap-1 rounded-xl border border-[#E6D6C6] bg-[#F6EFE7] p-1.5 shadow-inner">
            {NAV_ITEMS.map((item) => {
              const Icon = item.icon;
              const isActive = item.id === activeSection;
              return (
                <button
                  key={item.id}
                  type="button"
                  onClick={() => onSelectSection(item.id)}
                  className={cn(
                    "group relative flex items-center gap-2 rounded-lg px-3.5 py-1.5 text-xs font-semibold whitespace-nowrap transition-all duration-150",
                    isActive
                      ? "bg-[#6B3E26] text-[#FFFCF7] shadow-sm"
                      : "text-[#806D60] hover:bg-[#FFFCF7] hover:text-[#38271F]"
                  )}
                >
                  <Icon
                    className={cn(
                      "h-3.5 w-3.5 shrink-0 transition-transform group-hover:scale-105",
                      isActive ? "text-[#E9D5C0]" : "text-[#B98255]"
                    )}
                  />
                  <span>{item.label}</span>
                </button>
              );
            })}
          </nav>

          {/* Quick Controls & Status Pill */}
          <div className="flex items-center gap-2.5">
            {/* Active Run Selector */}
            {experiments.length > 0 && (
              <div className="hidden xl:flex items-center gap-1.5 rounded-lg border border-[#E6D6C6] bg-[#FFFCF7] px-2.5 py-1 shadow-card">
                <Layers className="h-3.5 w-3.5 text-[#B98255]" />
                <span className="text-[11px] font-semibold text-[#806D60]">
                  Run:
                </span>
                <select
                  value={selectedExperimentId || ""}
                  onChange={(e) => onSelectExperiment(e.target.value)}
                  className="bg-transparent font-medium text-xs text-[#38271F] outline-none cursor-pointer max-w-[140px] truncate"
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
                "hidden sm:flex items-center gap-1.5 rounded-lg border px-2.5 py-1 text-[11px] font-medium shadow-card",
                health
                  ? "border-[#B7D5C4] bg-[#EDF6F0] text-[#2D6A4F]"
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
                  health ? "bg-[#2D6A4F] animate-pulse" : "bg-amber-500"
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
              className="h-8 rounded-lg border-[#E6D6C6] bg-[#FFFCF7] hover:bg-[#FAF3EC] hover:border-[#B98255] text-xs font-semibold shadow-card transition-all"
              title="Sync metrics and status from backend"
            >
              <RefreshCw className="h-3.5 w-3.5 text-[#6B3E26]" />
              <span className="hidden md:inline">Sync</span>
            </Button>
          </div>
        </div>

        {/* Mobile/Tablet Horizontal Scroll Nav Tabs */}
        <div className="flex lg:hidden overflow-x-auto border-t border-[#E6D6C6] bg-[#FFFCF7] px-3 py-1.5 no-scrollbar gap-1">
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
                    ? "bg-[#6B3E26] text-[#FFFCF7] shadow-xs font-semibold"
                    : "text-[#806D60] hover:bg-[#FAF3EC] hover:text-[#38271F]"
                )}
              >
                <Icon className={cn("h-3.5 w-3.5", isActive ? "text-[#E9D5C0]" : "text-[#B98255]")} />
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
