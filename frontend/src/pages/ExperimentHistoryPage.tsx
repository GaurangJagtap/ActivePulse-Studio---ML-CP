import { useEffect, useState } from "react";
import { Download, Eye, GitCompare, Trash2 } from "lucide-react";
import { LearningCurveChart } from "@/charts/LearningCurveChart";
import { ErrorBanner } from "@/components/shared/FeedbackBanner";
import { StatusBadge } from "@/components/shared/StatusBadge";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { api } from "@/services/api";
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

interface ExperimentHistoryPageProps {
  experiments: ExperimentSummary[];
  selectedExperimentId: string | null;
  onSelectExperiment: (id: string) => void;
  onRefreshList: () => void;
  onNavigate: (section: NavSectionId) => void;
}

export function ExperimentHistoryPage({
  experiments,
  selectedExperimentId,
  onSelectExperiment,
  onRefreshList,
  onNavigate,
}: ExperimentHistoryPageProps) {
  const [compareIds, setCompareIds] = useState<string[]>([]);
  const [inspected, setInspected] = useState<ExperimentDetailResponse | null>(
    null
  );
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!selectedExperimentId) {
      setInspected(null);
      return;
    }
    api
      .getExperiment(selectedExperimentId)
      .then(setInspected)
      .catch(() => {});
  }, [selectedExperimentId]);

  function toggleCompare(id: string) {
    setCompareIds((prev) =>
      prev.includes(id)
        ? prev.filter((x) => x !== id)
        : prev.length < 4
          ? [...prev, id]
          : prev
    );
  }

  async function handleDelete(id: string) {
    setError(null);
    try {
      await api.deleteExperiment(id);
      setCompareIds((prev) => prev.filter((x) => x !== id));
      onRefreshList();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Delete failed.");
    }
  }

  const comparedRuns = experiments.filter((e) => compareIds.includes(e.id));

  return (
    <div className="space-y-6">
      {error && <ErrorBanner message={error} />}

      {/* Experiments Ledger */}
      <Card>
        <CardHeader className="flex flex-row items-center justify-between">
          <div>
            <CardTitle>Experiment History Ledger</CardTitle>
            <CardDescription>
              All Active Learning runs persisted in SQLite with full metrics and CSV exports
            </CardDescription>
          </div>
          <Badge variant="neutral">{experiments.length} Runs Recorded</Badge>
        </CardHeader>
        <CardContent>
          {experiments.length === 0 ? (
            <div className="py-10 text-center text-xs text-[#64748B]">
              No experiments recorded yet. Run a simulation in the Active Learning Lab to create one.
            </div>
          ) : (
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Compare</TableHead>
                  <TableHead>Experiment</TableHead>
                  <TableHead>Dataset / Model</TableHead>
                  <TableHead>Strategy</TableHead>
                  <TableHead>Rounds</TableHead>
                  <TableHead>Labeled</TableHead>
                  <TableHead>Accuracy</TableHead>
                  <TableHead>F1 / ECE</TableHead>
                  <TableHead>Status</TableHead>
                  <TableHead>Created</TableHead>
                  <TableHead className="text-right">Actions</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {experiments.map((exp) => {
                  const isSelected = exp.id === selectedExperimentId;
                  const isCompared = compareIds.includes(exp.id);
                  return (
                    <TableRow
                      key={exp.id}
                      className={isSelected ? "bg-[#EFF6FF]/60" : undefined}
                    >
                      <TableCell>
                        <input
                          type="checkbox"
                          checked={isCompared}
                          onChange={() => toggleCompare(exp.id)}
                          className="h-4 w-4 rounded border-[#CBD5E1] text-[#2563EB]"
                          title="Select up to 4 runs to compare"
                        />
                      </TableCell>
                      <TableCell>
                        <div className="font-medium text-[#172033]">
                          {exp.name}
                        </div>
                        <div className="font-mono text-[11px] text-[#64748B]">
                          {exp.id}
                        </div>
                      </TableCell>
                      <TableCell className="font-mono text-xs">
                        {exp.dataset} · {exp.model_type}
                      </TableCell>
                      <TableCell className="font-mono text-xs">
                        {exp.strategy}
                      </TableCell>
                      <TableCell className="font-mono text-xs">
                        {exp.current_round}/{exp.max_rounds}
                      </TableCell>
                      <TableCell className="font-mono text-xs">
                        {exp.n_labeled}
                      </TableCell>
                      <TableCell className="font-mono text-xs font-semibold text-[#2563EB]">
                        {formatPct(exp.latest_accuracy)}
                      </TableCell>
                      <TableCell className="font-mono text-xs text-[#64748B]">
                        {formatPct(exp.latest_f1)} / {formatNum(exp.latest_ece)}
                      </TableCell>
                      <TableCell>
                        <StatusBadge status={exp.status} />
                      </TableCell>
                      <TableCell className="text-xs text-[#64748B]">
                        {formatShortDate(exp.created_at)}
                      </TableCell>
                      <TableCell className="text-right">
                        <div className="flex justify-end gap-1">
                          <Button
                            variant="ghost"
                            size="icon"
                            title="Inspect experiment"
                            onClick={() => onSelectExperiment(exp.id)}
                          >
                            <Eye className="h-4 w-4" />
                          </Button>
                          <a
                            href={api.getExperimentExportUrl(exp.id)}
                            download={`${exp.id}_metrics.csv`}
                            className="inline-flex h-9 w-9 items-center justify-center rounded-lg text-[#64748B] hover:bg-[#F1F5F9] hover:text-[#172033]"
                            title="Export metrics CSV"
                          >
                            <Download className="h-4 w-4" />
                          </a>
                          <Button
                            variant="ghost"
                            size="icon"
                            title="Delete experiment"
                            onClick={() => handleDelete(exp.id)}
                          >
                            <Trash2 className="h-4 w-4 text-[#DC2626]" />
                          </Button>
                        </div>
                      </TableCell>
                    </TableRow>
                  );
                })}
              </TableBody>
            </Table>
          )}
        </CardContent>
      </Card>

      {/* Side-by-Side Comparison */}
      {comparedRuns.length >= 2 && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <GitCompare className="h-4 w-4 text-[#0D9488]" />
              <span>Side-by-Side Experiment Comparison ({comparedRuns.length})</span>
            </CardTitle>
          </CardHeader>
          <CardContent>
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Metric</TableHead>
                  {comparedRuns.map((r) => (
                    <TableHead key={r.id}>{r.name}</TableHead>
                  ))}
                </TableRow>
              </TableHeader>
              <TableBody>
                <TableRow>
                  <TableCell className="font-medium text-[#64748B]">Dataset / Model</TableCell>
                  {comparedRuns.map((r) => (
                    <TableCell key={r.id} className="font-mono text-xs">
                      {r.dataset} / {r.model_type}
                    </TableCell>
                  ))}
                </TableRow>
                <TableRow>
                  <TableCell className="font-medium text-[#64748B]">Strategy</TableCell>
                  {comparedRuns.map((r) => (
                    <TableCell key={r.id} className="font-mono text-xs font-semibold">
                      {r.strategy}
                    </TableCell>
                  ))}
                </TableRow>
                <TableRow>
                  <TableCell className="font-medium text-[#64748B]">Labeled Samples</TableCell>
                  {comparedRuns.map((r) => (
                    <TableCell key={r.id} className="font-mono text-xs">
                      {r.n_labeled}
                    </TableCell>
                  ))}
                </TableRow>
                <TableRow>
                  <TableCell className="font-medium text-[#64748B]">Holdout Accuracy</TableCell>
                  {comparedRuns.map((r) => (
                    <TableCell key={r.id} className="font-mono text-xs font-semibold text-[#2563EB]">
                      {formatPct(r.latest_accuracy)}
                    </TableCell>
                  ))}
                </TableRow>
                <TableRow>
                  <TableCell className="font-medium text-[#64748B]">Macro-F1</TableCell>
                  {comparedRuns.map((r) => (
                    <TableCell key={r.id} className="font-mono text-xs text-[#0D9488]">
                      {formatPct(r.latest_f1)}
                    </TableCell>
                  ))}
                </TableRow>
                <TableRow>
                  <TableCell className="font-medium text-[#64748B]">ECE</TableCell>
                  {comparedRuns.map((r) => (
                    <TableCell key={r.id} className="font-mono text-xs">
                      {formatNum(r.latest_ece)}
                    </TableCell>
                  ))}
                </TableRow>
              </TableBody>
            </Table>
          </CardContent>
        </Card>
      )}

      {/* Selected Experiment Detail Inspector */}
      {inspected && (
        <div className="grid grid-cols-1 gap-6 lg:grid-cols-3">
          <Card className="lg:col-span-2">
            <CardHeader className="flex flex-row items-center justify-between">
              <div>
                <CardTitle>{inspected.summary.name}</CardTitle>
                <CardDescription className="font-mono text-xs">
                  {inspected.summary.id} · {inspected.stop_reason || inspected.summary.current_stage}
                </CardDescription>
              </div>
              <Button
                variant="outline"
                size="sm"
                onClick={() => onNavigate("lab")}
              >
                Inspect in Lab
              </Button>
            </CardHeader>
            <CardContent>
              <LearningCurveChart rounds={inspected.rounds} height={280} />
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Run Configuration & Cost Summary</CardTitle>
            </CardHeader>
            <CardContent className="space-y-3 text-xs">
              {inspected.cost_savings && (
                <div className="rounded-lg border border-[#BBF7D0] bg-[#F0FDF4] p-3">
                  <div className="font-semibold text-[#15803D]">
                    Est. Savings:{" "}
                    {formatCurrency(
                      inspected.cost_savings.estimated_dollars_saved
                    )}
                  </div>
                  <div className="mt-0.5 text-[#166534]">
                    {inspected.cost_savings.estimated_labels_saved} fewer labels (
                    {inspected.cost_savings.pct_dataset_labeled}% dataset used)
                  </div>
                </div>
              )}
              <div className="max-h-52 space-y-1.5 overflow-y-auto rounded-lg border border-[#E2E8F0] bg-[#F8FAFC] p-3 font-mono text-[11px]">
                {Object.entries(inspected.config).map(([k, v]) => (
                  <div key={k} className="flex justify-between gap-2">
                    <span className="text-[#64748B]">{k}:</span>
                    <span className="truncate text-[#172033]">
                      {String(v)}
                    </span>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </div>
      )}
    </div>
  );
}

