import {
  Activity,
  ArrowRight,
  CheckCircle2,
  Coins,
  Database,
  Layers,
  Target,
} from "lucide-react";
import { LearningCurveChart } from "@/charts/LearningCurveChart";
import { MetricCard } from "@/components/shared/MetricCard";
import { PipelineTracker } from "@/components/shared/PipelineTracker";
import { StatusBadge } from "@/components/shared/StatusBadge";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import type {
  ExperimentDetailResponse,
  ExperimentSummary,
  NavSectionId,
} from "@/types/api";
import {
  formatCurrency,
  formatNum,
  formatPct,
  formatShortDate,
} from "@/utils/format";

interface OverviewDashboardPageProps {
  experiments: ExperimentSummary[];
  activeDetail: ExperimentDetailResponse | null;
  onNavigate: (section: NavSectionId) => void;
  onSelectExperiment: (id: string) => void;
}

export function OverviewDashboardPage({
  experiments,
  activeDetail,
  onNavigate,
  onSelectExperiment,
}: OverviewDashboardPageProps) {
  const summary = activeDetail?.summary;
  const savings = activeDetail?.cost_savings;
  const rounds = activeDetail?.rounds || [];
  const latestRound = rounds.length ? rounds[rounds.length - 1] : null;
  const firstRound = rounds.length ? rounds[0] : null;

  const accDelta =
    latestRound && firstRound && rounds.length > 1
      ? `+${((latestRound.test_acc - firstRound.test_acc) * 100).toFixed(1)}%`
      : undefined;

  const cfgBudget = Number(activeDetail?.config?.label_budget ?? 0);
  const cfgMaxSamples = Number(activeDetail?.config?.max_samples ?? 0);
  const budgetDenom = cfgBudget > 0 ? cfgBudget : cfgMaxSamples || 1;
  const budgetPct = summary
    ? Math.min(100, (summary.n_labeled / budgetDenom) * 100)
    : 0;

  return (
    <div className="space-y-6">
      {/* Active Experiment Status Highlight / Onboarding Banner */}
      {summary ? (
        <Card tone="blue">
          <CardContent className="flex flex-col items-start justify-between gap-4 p-5 sm:flex-row sm:items-center">
            <div className="space-y-1">
              <div className="flex flex-wrap items-center gap-2.5">
                <h2 className="text-base font-semibold text-[#38271F]">
                  {summary.name}
                </h2>
                <StatusBadge status={summary.status} />
              </div>
            </div>
            <div className="flex flex-wrap items-center gap-2">
              {summary.status === "awaiting_annotations" && (
                <Button
                  size="sm"
                  variant="teal"
                  onClick={() => onNavigate("annotation")}
                >
                  Open Label Queue ({summary.n_queued})
                  <ArrowRight className="h-3.5 w-3.5" />
                </Button>
              )}
              <Button
                size="sm"
                variant="outline"
                onClick={() => onNavigate("lab")}
              >
                Inspect in AL Lab
              </Button>
            </div>
          </CardContent>
        </Card>
      ) : (
        <Card tone="blue">
          <CardContent className="flex flex-col items-start justify-between gap-4 p-5 sm:flex-row sm:items-center">
            <div>
              <h2 className="text-sm font-semibold text-[#263452]">
                No Active Learning runs recorded yet
              </h2>
              <p className="mt-1 text-xs text-[#806D60]">
                Launch an experiment in the Active Learning Lab to train a PyTorch model, evaluate holdout accuracy, and compute real label-cost savings.
              </p>
            </div>
            <Button onClick={() => onNavigate("lab")}>
              Open Active Learning Lab
              <ArrowRight className="h-4 w-4" />
            </Button>
          </CardContent>
        </Card>
      )}

      {/* Primary KPI Grid */}
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 xl:grid-cols-4">
        <MetricCard
          title="Holdout Test Accuracy"
          value={formatPct(summary?.latest_accuracy)}
          delta={accDelta}
          subtitle={
            summary
              ? `Macro-F1: ${formatPct(summary.latest_f1)} · ECE: ${formatNum(summary.latest_ece)}`
              : "Zero-leakage test partition"
          }
          icon={Target}
          accent="blue"
          tone="default"
        />
        <MetricCard
          title="Labeled vs Unlabeled"
          value={summary ? `${summary.n_labeled}` : "—"}
          subtitle={
            summary
              ? `${summary.n_pool} unlabeled pool · ${summary.n_queued} queued`
              : "Pool distribution"
          }
          icon={Database}
          accent="teal"
          tone="mint"
        />
        <MetricCard
          title="Current AL Round"
          value={
            summary ? `${summary.current_round} / ${summary.max_rounds}` : "—"
          }
          subtitle={
            summary
              ? `${summary.strategy.toUpperCase()} on ${summary.dataset.toUpperCase()}`
              : "Active learning iteration"
          }
          icon={Layers}
          accent="lavender"
          tone="lavender"
        />
        <MetricCard
          title="Est. Cost Reduction"
          value={savings ? formatCurrency(savings.estimated_dollars_saved) : "—"}
          subtitle={
            savings
              ? `${savings.estimated_labels_saved} labels · ~${savings.estimated_hours_saved} hrs saved`
              : "Calculated vs 75% passive baseline"
          }
          icon={Coins}
          accent="green"
          tone="secondary"
        />
      </div>

      {/* Pipeline Stage Tracker & Budget Utilization */}
      <div className="grid grid-cols-1 gap-4 lg:grid-cols-3">
        <div className="lg:col-span-2">
          <div className="mb-2 flex items-center justify-between">
            <h2 className="text-xs font-semibold uppercase tracking-wider text-[#806D60]">
              Active Learning Pipeline Progress
            </h2>
            {summary && <StatusBadge status={summary.status} />}
          </div>
          <PipelineTracker
            activeIndex={
              summary?.status === "completed"
                ? 5
                : summary?.current_stage.includes("Scoring") ||
                    summary?.current_stage.includes("Awaiting")
                  ? 3
                  : summary?.current_stage.includes("Retraining")
                    ? 1
                    : 0
            }
            isRunning={summary?.status === "running"}
            isCompleted={summary?.status === "completed"}
          />
        </div>

        <Card tone="lavender">
          <CardHeader className="pb-2">
            <CardTitle className="flex items-center justify-between text-sm">
              <span>Annotation Budget Utilization</span>
              <span className="font-mono text-xs text-[#4568D9]">
                {budgetPct.toFixed(1)}%
              </span>
            </CardTitle>
            <CardDescription>
              {summary
                ? `${summary.n_labeled} of ${budgetDenom} budget samples labeled`
                : "No active budget"}
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-3">
            <Progress value={budgetPct} />
            <div className="flex items-center justify-between text-xs text-[#806D60]">
              <span>
                Dataset labeled:{" "}
                <strong className="font-mono text-[#263452]">
                  {savings ? `${savings.pct_dataset_labeled}%` : "—"}
                </strong>
              </span>
              {summary?.status === "awaiting_annotations" && (
                <Button
                  size="sm"
                  variant="teal"
                  onClick={() => onNavigate("annotation")}
                >
                  Label Queue ({summary.n_queued})
                </Button>
              )}
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Learning Curve + Summary */}
      <div className="grid grid-cols-1 gap-6 lg:grid-cols-3">
        <Card className="lg:col-span-2">
          <CardHeader className="flex flex-row items-center justify-between">
            <div>
              <CardTitle>Model Learning Curve</CardTitle>
              <CardDescription>
                Holdout test accuracy & Macro-F1 progression across labeled samples
              </CardDescription>
            </div>
            {summary && (
              <span className="rounded-lg bg-[#FAF3EC] border border-[#D8BEA6] px-2.5 py-1 font-mono text-xs font-medium text-[#6B3E26]">
                {summary.model_type}
              </span>
            )}
          </CardHeader>
          <CardContent>
            <LearningCurveChart rounds={rounds} height={360} />
          </CardContent>
        </Card>

        <Card tone="mint">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Activity className="h-4 w-4 text-[#B98255]" />
              <span>Active Run Telemetry</span>
            </CardTitle>
            <CardDescription>
              Latest round metrics from SQLite state
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-3">
            {latestRound ? (
              <div className="space-y-2.5 text-sm">
                <div className="flex justify-between border-b border-[#E6D6C6] pb-2">
                  <span className="text-[#806D60]">Strategy</span>
                  <span className="font-mono font-medium text-[#38271F]">
                    {latestRound.strategy}
                  </span>
                </div>
                <div className="flex justify-between border-b border-[#E6D6C6] pb-2">
                  <span className="text-[#806D60]">Holdout Accuracy</span>
                  <span className="font-mono font-semibold text-[#6B3E26]">
                    {formatPct(latestRound.test_acc)}
                  </span>
                </div>
                <div className="flex justify-between border-b border-[#E6D6C6] pb-2">
                  <span className="text-[#806D60]">Macro-F1 Score</span>
                  <span className="font-mono font-medium text-[#B98255]">
                    {formatPct(latestRound.test_f1)}
                  </span>
                </div>
                <div className="flex justify-between border-b border-[#E6D6C6] pb-2">
                  <span className="text-[#806D60]">Calibration Error (ECE)</span>
                  <span className="font-mono font-medium text-[#38271F]">
                    {formatNum(latestRound.ece)}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-[#806D60]">Learned Temperature (T)</span>
                  <span className="font-mono font-medium text-[#38271F]">
                    {formatNum(latestRound.temperature, 2)}
                  </span>
                </div>
                {savings && (
                  <p className="rounded-lg bg-[#FAF3EC] border border-[#E6D6C6] p-3 text-[11px] leading-relaxed text-[#806D60]">
                    {savings.methodology_note}
                  </p>
                )}
              </div>
            ) : (
              <div className="py-10 text-center text-xs text-[#806D60]">
                Start a simulation or human-in-the-loop run to populate round telemetry.
              </div>
            )}
          </CardContent>
        </Card>
      </div>

      {/* Recent Experiment Activity */}
      <Card tone="secondary">
        <CardHeader className="flex flex-row items-center justify-between">
          <div>
            <CardTitle>Recent Experiment Activity</CardTitle>
            <CardDescription>
              Persisted Active Learning runs stored in local SQLite workspaces
            </CardDescription>
          </div>
          <Button variant="outline" size="sm" onClick={() => onNavigate("history")}>
            View All Experiments
          </Button>
        </CardHeader>
        <CardContent>
          {experiments.length === 0 ? (
            <div className="flex items-center justify-center gap-2 py-8 text-xs text-[#806D60]">
              <CheckCircle2 className="h-4 w-4 text-[#718BE0]" />
              <span>No experiments yet. Click "Active Learning Lab" to run your first loop.</span>
            </div>
          ) : (
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Experiment</TableHead>
                  <TableHead>Dataset</TableHead>
                  <TableHead>Strategy</TableHead>
                  <TableHead>Mode</TableHead>
                  <TableHead>Round</TableHead>
                  <TableHead>Accuracy</TableHead>
                  <TableHead>Status</TableHead>
                  <TableHead>Updated</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {experiments.slice(0, 6).map((exp) => (
                  <TableRow
                    key={exp.id}
                    className="cursor-pointer"
                    onClick={() => onSelectExperiment(exp.id)}
                  >
                    <TableCell className="font-medium">
                      <div>{exp.name}</div>
                      <div className="font-mono text-[11px] text-[#806D60]">
                        {exp.id}
                      </div>
                    </TableCell>
                    <TableCell className="font-mono text-xs uppercase">
                      {exp.dataset}
                    </TableCell>
                    <TableCell className="font-mono text-xs">
                      {exp.strategy}
                    </TableCell>
                    <TableCell className="text-xs text-[#806D60]">
                      {exp.mode === "human_in_the_loop"
                        ? "Human-in-the-Loop"
                        : "Simulated Oracle"}
                    </TableCell>
                    <TableCell className="font-mono text-xs">
                      {exp.current_round}/{exp.max_rounds}
                    </TableCell>
                    <TableCell className="font-mono text-xs font-semibold text-[#6B3E26]">
                      {formatPct(exp.latest_accuracy)}
                    </TableCell>
                    <TableCell>
                      <StatusBadge status={exp.status} />
                    </TableCell>
                    <TableCell className="text-xs text-[#806D60]">
                      {formatShortDate(exp.updated_at)}
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          )}
        </CardContent>
      </Card>
    </div>
  );
}

