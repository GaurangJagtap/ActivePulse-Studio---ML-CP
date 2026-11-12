import { useState } from "react";
import {
  BarChart3,
  ChevronLeft,
  ChevronRight,
  Database,
  FlaskConical,
  History,
  LayoutDashboard,
  PenTool,
  RefreshCw,
} from "lucide-react";
import { cn } from "@/lib/utils";
import { Badge } from "@/components/ui/badge";
import { Select } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import type {
  ExperimentSummary,
  HealthResponse,
  NavSectionId,
} from "@/types/api";

interface NavItem {
  id: NavSectionId;
  label: string;
  subtitle: string;
  icon: typeof LayoutDashboard;
}

export const NAV_ITEMS: NavItem[] = [
  {
    id: "overview",
    label: "Overview Dashboard",
    subtitle: "Executive metrics, learning curve & pipeline state",
    icon: LayoutDashboard,
  },
  {
    id: "datasets",
    label: "Dataset Explorer",
    subtitle: "Splits, class balance & sample inspection",
    icon: Database,
  },
  {
    id: "lab",
    label: "Active Learning Lab",
    subtitle: "Configure, run & inspect uncertainty/PCA loops",
    icon: FlaskConical,
  },
  {
    id: "annotation",
    label: "Annotation Studio",
    subtitle: "Human-in-the-loop labeling queue & QA",
    icon: PenTool,
  },
  {
    id: "benchmarks",
    label: "Benchmark Arena",
    subtitle: "Multi-seed strategy vs random baseline savings",
    icon: BarChart3,
  },
  {
    id: "history",
    label: "Experiment History",
    subtitle: "Run ledger, comparison & CSV exports",
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
  const [collapsed, setCollapsed] = useState(false);
  const currentNav =
    NAV_ITEMS.find((n) => n.id === activeSection) || NAV_ITEMS[0];

  return (
    <div className="flex min-h-screen bg-[#F7F8FC] text-[#172033]">
      {/* Persistent Dark Navy Sidebar */}
      <aside
        className={cn(
          "sticky top-0 flex h-screen shrink-0 flex-col bg-[#101827] text-slate-200 transition-all duration-200",
          collapsed ? "w-[72px]" : "w-64"
        )}
      >
        {/* Brand Header */}
        <div className="flex h-16 items-center justify-between border-b border-slate-800 px-4">
          <div className="flex items-center gap-3 overflow-hidden">
            {/* Custom SVG Logo */}
            <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-lg bg-[#2563EB] text-white">
              <svg
                width="20"
                height="20"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="2.2"
                strokeLinecap="round"
                strokeLinejoin="round"
              >
                <polyline points="22 12 18 12 15 20 9 4 6 12 2 12" />
              </svg>
            </div>
            {!collapsed && (
              <div className="min-w-0">
                <div className="truncate text-sm font-semibold tracking-tight text-white">
                  ActivePulse
                </div>
                <div className="truncate text-[11px] text-slate-400">
                  ML Studio v1.0
                </div>
              </div>
            )}
          </div>
          <button
            type="button"
            onClick={() => setCollapsed((v) => !v)}
            className="rounded-md p-1 text-slate-400 hover:bg-slate-800 hover:text-white"
            title={collapsed ? "Expand sidebar" : "Collapse sidebar"}
          >
            {collapsed ? (
              <ChevronRight className="h-4 w-4" />
            ) : (
              <ChevronLeft className="h-4 w-4" />
            )}
          </button>
        </div>

        {/* Navigation Links */}
        <nav className="flex-1 space-y-1 overflow-y-auto px-3 py-4">
          {!collapsed && (
            <div className="px-2 pb-2 text-[10px] font-semibold uppercase tracking-wider text-slate-500">
              Platform Workspaces
            </div>
          )}
          {NAV_ITEMS.map((item) => {
            const Icon = item.icon;
            const isActive = item.id === activeSection;
            return (
              <button
                key={item.id}
                type="button"
                onClick={() => onSelectSection(item.id)}
                title={collapsed ? item.label : undefined}
                className={cn(
                  "flex w-full items-center gap-3 rounded-lg px-3 py-2.5 text-left text-sm font-medium transition-colors",
                  isActive
                    ? "bg-[#2563EB] text-white"
                    : "text-slate-300 hover:bg-slate-800/70 hover:text-white"
                )}
              >
                <Icon className="h-4 w-4 shrink-0" />
                {!collapsed && <span className="truncate">{item.label}</span>}
              </button>
            );
          })}
        </nav>

        {/* Bottom Profile & Engine Area */}
        <div className="border-t border-slate-800 p-3">
          {collapsed ? (
            <div
              className="mx-auto flex h-9 w-9 items-center justify-center rounded-full bg-slate-800 font-mono text-xs font-semibold text-teal-400"
              title={health ? `PyTorch ${health.pytorch_version}` : "Connecting..."}
            >
              ML
            </div>
          ) : (
            <div className="rounded-lg bg-slate-900/90 p-3">
              <div className="flex items-center justify-between">
                <span className="text-xs font-medium text-white">
                  ML Workspace
                </span>
                <span
                  className={cn(
                    "h-2 w-2 rounded-full",
                    health ? "bg-[#16A34A]" : "bg-amber-400"
                  )}
                />
              </div>
              <p className="mt-1 font-mono text-[11px] text-slate-400">
                {health
                  ? `PyTorch ${health.pytorch_version} · ${health.device.toUpperCase()}`
                  : "Connecting to FastAPI..."}
              </p>
              <p className="mt-0.5 text-[11px] text-slate-500">
                {health
                  ? `${health.registered_strategies} strategies loaded`
                  : "Checking backend status"}
              </p>
            </div>
          )}
        </div>
      </aside>

      {/* Main Content Shell */}
      <div className="flex min-w-0 flex-1 flex-col">
        {/* Top Navigation Bar */}
        <header className="sticky top-0 z-20 flex h-16 items-center justify-between gap-4 border-b border-[#E2E8F0] bg-white px-6">
          <div className="min-w-0">
            <h1 className="truncate text-base font-semibold text-[#172033]">
              {currentNav.label}
            </h1>
            <p className="truncate text-xs text-[#64748B]">
              {currentNav.subtitle}
            </p>
          </div>

          <div className="flex items-center gap-3">
            {experiments.length > 0 && (
              <div className="hidden items-center gap-2 md:flex">
                <span className="text-xs font-medium text-[#64748B]">
                  Active Run:
                </span>
                <Select
                  value={selectedExperimentId || ""}
                  onChange={(e) => onSelectExperiment(e.target.value)}
                  className="h-8 w-64 text-xs"
                >
                  {experiments.map((exp) => (
                    <option key={exp.id} value={exp.id}>
                      {exp.name} [{exp.status}]
                    </option>
                  ))}
                </Select>
              </div>
            )}

            <Badge variant={health ? "success" : "warning"}>
              <span
                className={cn(
                  "h-1.5 w-1.5 rounded-full",
                  health ? "bg-[#16A34A]" : "bg-amber-500"
                )}
              />
              {health ? "FastAPI Connected" : "API Offline"}
            </Badge>

            <Button
              variant="outline"
              size="sm"
              onClick={onRefreshAll}
              title="Refresh data from backend"
            >
              <RefreshCw className="h-3.5 w-3.5" />
              <span className="hidden sm:inline">Refresh</span>
            </Button>
          </div>
        </header>

        {/* Responsive Content Canvas */}
        <main className="mx-auto w-full max-w-[1440px] flex-1 p-6">
          {children}
        </main>
      </div>
    </div>
  );
}

