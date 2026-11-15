import { useEffect, useState } from "react";
import { Award, Download, Play, Sparkles, Trophy } from "lucide-react";
import {
  BenchmarkAlcBarChart,
  BenchmarkCurvesChart,
} from "@/charts/BenchmarkCharts";
import { ErrorBanner, LoadingState } from "@/components/shared/FeedbackBanner";
import { MetricCard } from "@/components/shared/MetricCard";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Input, Select } from "@/components/ui/input";
import { Progress } from "@/components/ui/progress";
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
  BenchmarkJobStatus,
  BenchmarkResultsResponse,
  StrategyInfo,
} from "@/types/api";
import { formatNum, formatPct } from "@/utils/format";

export function BenchmarkArenaPage({
  strategies,
}: {
  strategies: StrategyInfo[];
}) {
  const [results, setResults] = useState<BenchmarkResultsResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Benchmark runner config
  const [dataset, setDataset] = useState<"cifar10" | "agnews">("cifar10");
  const [selectedStrategies, setSelectedStrategies] = useState<string[]>([
    "random",
    "entropy",
    "bald",
    "coreset",
    "hybrid",
  ]);
  const [seedsText, setSeedsText] = useState("42, 43");
  const [rounds, setRounds] = useState(3);
  const [batchSize, setBatchSize] = useState(15);
  const [targetAcc, setTargetAcc] = useState(0.25);

  const [job, setJob] = useState<BenchmarkJobStatus | null>(null);
  const [plotKey, setPlotKey] = useState<number>(Date.now());

  async function loadResults() {
    setLoading(true);
    setError(null);
    try {
      const data = await api.getBenchmarkResults();
      setResults(data);
      setPlotKey(Date.now());
    } catch (err) {
      setError(
        err instanceof Error ? err.message : "Failed to load benchmark results."
      );
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    loadResults();
  }, []);

  useEffect(() => {
    if (!job || job.status !== "running") return;
    const timer = window.setInterval(async () => {
      try {
        const next = await api.getBenchmarkStatus(job.job_id);
        setJob(next);
        if (next.status !== "running") {
          loadResults();
        }
      } catch {
        // ignore polling blip
      }
    }, 1200);
    return () => window.clearInterval(timer);
  }, [job]);

  function toggleStrategy(id: string) {
    if (id === "random") return; // Keep random baseline selected
    setSelectedStrategies((prev) =>
      prev.includes(id) ? prev.filter((x) => x !== id) : [...prev, id]
    );
  }

  async function handleRunBenchmark(e: React.FormEvent) {
    e.preventDefault();
    setError(null);
    const parsedSeeds = seedsText
      .split(",")
      .map((s) => Number(s.trim()))
      .filter((n) => !Number.isNaN(n));
    if (!parsedSeeds.length) {
      setError("Provide at least one integer random seed (e.g., 42, 43).");
      return;
    }

    try {
      const started = await api.runBenchmark({
        dataset,
        strategies: selectedStrategies,
        seeds: parsedSeeds.slice(0, 5),
        max_samples: 450,
        initial_size: 30,
        batch_size: batchSize,
        rounds,
        epochs: 4,
        target_accuracy: targetAcc,
        use_synthetic: true,
      });
      setJob(started);
    } catch (err) {
      setError(
        err instanceof Error ? err.message : "Failed to start benchmark."
      );
    }
  }

  const benchmarkStrategies = strategies.filter(
    (s) => s.id !== "cold_start_kmeans"
  );

  const summaryRows = results?.summary_rows || [];
  const bestAlcRow = summaryRows.length
    ? summaryRows.reduce((best, r) => (r.alc_mean > best.alc_mean ? r : best), summaryRows[0])
    : null;
  const bestSavingsRow = summaryRows.length
    ? summaryRows.reduce(
        (best, r) =>
          (r.label_reduction_pct ?? -Infinity) > (best.label_reduction_pct ?? -Infinity)
            ? r
            : best,
        summaryRows[0]
      )
    : null;

  return (
    <div className="space-y-6">
      {error && <ErrorBanner message={error} />}

      {/* Multi-Seed Benchmark Launcher */}
      <Card tone="secondary">
        <CardHeader className="flex flex-col justify-between gap-2 sm:flex-row sm:items-center">
          <div>
            <CardTitle className="flex items-center gap-2">
              <Trophy className="h-4 w-4 text-[#6B3E26]" />
              <span>Run Multi-Seed Benchmark Suite</span>
            </CardTitle>
            <CardDescription>
              Compare Active Learning strategies head-to-head against Uniform Random using src/eval/savings.py
            </CardDescription>
          </div>
          <a
            href={api.getBenchmarkExportUrl()}
            download="benchmark_savings.csv"
            className="inline-flex h-8 items-center gap-1.5 rounded-lg border border-[#E6D6C6] bg-[#FFFFFF] px-3 text-xs font-medium text-[#38271F] transition-colors hover:bg-[#FAF3EC] hover:border-[#6B3E26]"
          >
            <Download className="h-3.5 w-3.5" />
            Export CSV
          </a>
        </CardHeader>
        <CardContent>
          <form onSubmit={handleRunBenchmark} className="space-y-4">
            <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
              <div className="space-y-1.5">
                <label className="text-xs font-medium text-[#806D60]">
                  Dataset
                </label>
                <Select
                  value={dataset}
                  onChange={(e) =>
                    setDataset(e.target.value as "cifar10" | "agnews")
                  }
                >
                  <option value="cifar10">CIFAR-10 (Synthetic CPU)</option>
                  <option value="agnews">AG News (Embeddings)</option>
                </Select>
              </div>
              <div className="space-y-1.5">
                <label className="text-xs font-medium text-[#806D60]">
                  Seeds (comma-separated)
                </label>
                <Input
                  value={seedsText}
                  onChange={(e) => setSeedsText(e.target.value)}
                  placeholder="42, 43, 44"
                />
              </div>
              <div className="space-y-1.5">
                <label className="text-xs font-medium text-[#806D60]">
                  Rounds / Batch Size
                </label>
                <div className="grid grid-cols-2 gap-2">
                  <Input
                    type="number"
                    min={1}
                    max={8}
                    value={rounds}
                    onChange={(e) => setRounds(Number(e.target.value))}
                  />
                  <Input
                    type="number"
                    min={5}
                    max={40}
                    value={batchSize}
                    onChange={(e) => setBatchSize(Number(e.target.value))}
                  />
                </div>
              </div>
              <div className="space-y-1.5">
                <label className="text-xs font-medium text-[#806D60]">
                  Target Accuracy (0.05–1.0)
                </label>
                <Input
                  type="number"
                  step="0.05"
                  min={0.05}
                  max={1.0}
                  value={targetAcc}
                  onChange={(e) => setTargetAcc(Number(e.target.value))}
                />
              </div>
            </div>

            <div className="flex flex-wrap items-center justify-between gap-4 border-t border-[#E6D6C6] pt-4">
              <div className="flex flex-wrap gap-2">
                {benchmarkStrategies.map((s) => {
                  const active = selectedStrategies.includes(s.id);
                  return (
                    <button
                      key={s.id}
                      type="button"
                      onClick={() => toggleStrategy(s.id)}
                      className={`rounded-full border px-3 py-1 text-xs font-medium transition-colors ${
                        active
                          ? "border-[#6B3E26] bg-[#FAF3EC] text-[#6B3E26]"
                          : "border-[#E6D6C6] bg-[#FFFFFF] text-[#806D60] hover:border-[#6B3E26]"
                      }`}
                    >
                      {s.id}
                      {s.id === "random" ? " (baseline)" : ""}
                    </button>
                  );
                })}
              </div>

              <Button
                type="submit"
                disabled={job?.status === "running"}
              >
                <Play className="h-4 w-4" />
                {job?.status === "running"
                  ? `Running ${job.current_strategy} (seed ${job.current_seed})...`
                  : "Run Multi-Seed Benchmark"}
              </Button>
            </div>

            {job && job.status === "running" && (
              <div className="space-y-1.5 pt-2">
                <div className="flex justify-between text-xs text-[#806D60]">
                  <span>
                    Executing {job.current_strategy} · seed {job.current_seed}
                  </span>
                  <span className="font-mono text-[#38271F]">
                    {job.completed_steps} / {job.total_steps} runs
                  </span>
                </div>
                <Progress
                  value={
                    job.total_steps > 0
                      ? (job.completed_steps / job.total_steps) * 100
                      : 0
                  }
                />
              </div>
            )}
            {job?.status === "failed" && job.error_message && (
              <ErrorBanner message={job.error_message} />
            )}
          </form>
        </CardContent>
      </Card>

      {loading && !results ? (
        <LoadingState label="Loading benchmark metrics..." />
      ) : (
        results && (
          <>
            {/* Strategy Performance Highlights */}
            {bestAlcRow && (
              <div className="grid grid-cols-1 gap-4 md:grid-cols-3">
                <MetricCard
                  title="Top ALC Strategy"
                  value={bestAlcRow.strategy}
                  subtitle={`ALC Score: ${bestAlcRow.alc_display}`}
                  icon={Trophy}
                  accent="blue"
                  tone="blue"
                />
                <MetricCard
                  title="Highest Label Reduction"
                  value={
                    bestSavingsRow?.label_reduction_pct !== null &&
                    bestSavingsRow?.label_reduction_pct !== undefined
                      ? `${bestSavingsRow.label_reduction_pct}%`
                      : "—"
                  }
                  subtitle={
                    bestSavingsRow
                      ? `Strategy: ${bestSavingsRow.strategy} · ${bestSavingsRow.labels_to_target_display} labels`
                      : "Relative to Random baseline"
                  }
                  icon={Sparkles}
                  accent="teal"
                  tone="mint"
                />
                <MetricCard
                  title="Best Final Accuracy"
                  value={formatPct(
                    summaryRows.reduce(
                      (m, r) => Math.max(m, r.final_accuracy ?? 0),
                      0
                    )
                  )}
                  subtitle={`${summaryRows.length} strategies benchmarked`}
                  icon={Award}
                  accent="lavender"
                  tone="lavender"
                />
              </div>
            )}

            {/* Prominent Accuracy vs Budget + ALC Comparison */}
            <div className="grid grid-cols-1 gap-6 xl:grid-cols-12">
              <Card className="xl:col-span-7">
                <CardHeader>
                  <CardTitle>Accuracy vs Annotation Budget</CardTitle>
                  <CardDescription>
                    Mean holdout accuracy across seeds as labeled samples grow (dashed = Random baseline)
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <BenchmarkCurvesChart points={results.curve_points} height={380} />
                </CardContent>
              </Card>

              <Card tone="lavender" className="xl:col-span-5">
                <CardHeader>
                  <CardTitle>Area Under Learning Curve & Savings</CardTitle>
                  <CardDescription>
                    Source: <code className="font-mono">{results.source}</code>
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  {results.summary_rows.length ? (
                    <BenchmarkAlcBarChart rows={results.summary_rows} />
                  ) : (
                    <div className="py-12 text-center text-xs text-[#806D60]">
                      No benchmark summary rows available yet.
                    </div>
                  )}
                </CardContent>
              </Card>
            </div>

            {/* Strategy Comparison Table */}
            <Card tone="secondary">
              <CardHeader>
                <CardTitle>Strategy Efficiency & Calibration Comparison</CardTitle>
                <CardDescription>
                  Multi-seed mean ± 95% confidence intervals computed via src/eval/savings.py
                </CardDescription>
              </CardHeader>
              <CardContent>
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead>Strategy</TableHead>
                      <TableHead>Seeds</TableHead>
                      <TableHead>ALC (Mean ± 95% CI)</TableHead>
                      <TableHead>Labels to Target</TableHead>
                      <TableHead>% Reduction vs Random</TableHead>
                      <TableHead>Final Acc</TableHead>
                      <TableHead>Final F1</TableHead>
                      <TableHead>Final ECE</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {results.summary_rows.map((row) => {
                      const isTopAlc = bestAlcRow?.strategy === row.strategy;
                      return (
                        <TableRow
                          key={row.strategy}
                          className={isTopAlc ? "bg-[#FAF3EC]" : undefined}
                        >
                          <TableCell className="font-mono text-xs font-semibold">
                            {row.strategy}
                            {row.strategy === "random" && (
                              <Badge variant="neutral" className="ml-2">
                                Baseline
                              </Badge>
                            )}
                            {isTopAlc && (
                              <Badge variant="teal" className="ml-2">
                                Best ALC
                              </Badge>
                            )}
                          </TableCell>
                          <TableCell className="font-mono text-xs">
                            {row.seeds}
                          </TableCell>
                          <TableCell className="font-mono text-xs">
                            {row.alc_display}
                          </TableCell>
                          <TableCell className="font-mono text-xs">
                            {row.labels_to_target_display}
                          </TableCell>
                          <TableCell className="font-mono text-xs font-semibold text-[#B98255]">
                            {row.label_reduction_display}
                          </TableCell>
                          <TableCell className="font-mono text-xs font-medium text-[#6B3E26]">
                            {formatPct(row.final_accuracy)}
                          </TableCell>
                          <TableCell className="font-mono text-xs">
                            {formatPct(row.final_f1)}
                          </TableCell>
                          <TableCell className="font-mono text-xs">
                            {formatNum(row.final_ece)}
                          </TableCell>
                        </TableRow>
                      );
                    })}
                  </TableBody>
                </Table>
              </CardContent>
            </Card>

            {/* Saved Plot Artifact */}
            {results.has_learning_curve_image && (
              <Card>
                <CardHeader>
                  <CardTitle>Generated Confidence-Interval Plot Artifact</CardTitle>
                  <CardDescription>
                    Static high-resolution plot from results/learning_curves.png
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <img
                    src={`${api.getBenchmarkPlotUrl()}?t=${plotKey}`}
                    alt="Learning Curves with 95% CI"
                    className="mx-auto max-h-[420px] rounded-xl border border-[#D5DDF2]"
                  />
                </CardContent>
              </Card>
            )}
          </>
        )
      )}
    </div>
  );
}

