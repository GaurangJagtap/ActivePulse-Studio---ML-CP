import { useEffect, useState } from "react";
import { Play, Sparkles, Terminal } from "lucide-react";
import { ClassProbabilityChart } from "@/charts/ClassProbabilityChart";
import { LearningCurveChart } from "@/charts/LearningCurveChart";
import { PcaScatterPlot } from "@/charts/PcaScatterPlot";
import { ErrorBanner } from "@/components/shared/FeedbackBanner";
import { PipelineTracker } from "@/components/shared/PipelineTracker";
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
import { Input, Select } from "@/components/ui/input";
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
  DatasetSummary,
  ExperimentCreateRequest,
  ExperimentDetailResponse,
  ExperimentMode,
  ExperimentProgressResponse,
  ModelInfo,
  NavSectionId,
  PcaEmbeddingResponse,
  SelectedSamplesResponse,
  StrategyInfo,
} from "@/types/api";
import { formatNum, formatPct } from "@/utils/format";

interface ActiveLearningLabPageProps {
  datasets?: DatasetSummary[];
  models: ModelInfo[];
  strategies: StrategyInfo[];
  selectedExperimentId: string | null;
  activeDetail: ExperimentDetailResponse | null;
  onExperimentCreated: (id: string) => void;
  onRefreshExperiment: (id: string) => void;
  onNavigate: (section: NavSectionId) => void;
}

export function ActiveLearningLabPage({
  datasets = [],
  models,
  strategies,
  selectedExperimentId,
  activeDetail,
  onExperimentCreated,
  onRefreshExperiment,
  onNavigate,
}: ActiveLearningLabPageProps) {
  const [name, setName] = useState("");
  const [dataset, setDataset] = useState<string>("cifar10");
  const [modelType, setModelType] = useState<string>("cifar_cnn");
  const [strategy, setStrategy] = useState<string>("hybrid");
  const [mode, setMode] = useState<ExperimentMode>("simulated_oracle");
  const [initialSize, setInitialSize] = useState<number>(30);
  const [batchSize, setBatchSize] = useState<number>(15);
  const [maxRounds, setMaxRounds] = useState<number>(3);
  const [labelBudget, setLabelBudget] = useState<number>(200);
  const [maxSamples, setMaxSamples] = useState<number>(450);
  const [maxEpochs, setMaxEpochs] = useState<number>(6);
  const [lr, setLr] = useState<number>(0.01);
  const [seed, setSeed] = useState<number>(42);
  const [useSynthetic, setUseSynthetic] = useState<boolean>(true);
  const [noiseRate, setNoiseRate] = useState<number>(0.0);

  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [progress, setProgress] = useState<ExperimentProgressResponse | null>(null);
  const [pca, setPca] = useState<PcaEmbeddingResponse | null>(null);
  const [selected, setSelected] = useState<SelectedSamplesResponse | null>(null);

  const activeDatasetObj = datasets.find((d) => d.id === dataset);
  const compatibleModels = models.filter((m) => {
    if (m.supported_datasets.includes(dataset)) return true;
    if (activeDatasetObj?.supported_models?.includes(m.id)) return true;
    return false;
  });
  const activeStrategyInfo = strategies.find((s) => s.id === strategy);

  // Keep model compatible when switching dataset
  useEffect(() => {
    if (activeDatasetObj) {
      if (!compatibleModels.some((m) => m.id === modelType)) {
        const nextModel = activeDatasetObj.default_model || compatibleModels[0]?.id || "cifar_cnn";
        setModelType(nextModel);
      }
    } else {
      if (dataset === "cifar10" && modelType === "text_mlp") {
        setModelType("cifar_cnn");
        setLr(0.01);
      } else if (dataset === "agnews" && modelType !== "text_mlp") {
        setModelType("text_mlp");
        setLr(0.02);
      }
    }
  }, [dataset, modelType, activeDatasetObj, compatibleModels]);

  // Fetch PCA and selected samples whenever selected experiment changes or finishes
  useEffect(() => {
    if (!selectedExperimentId) {
      setPca(null);
      setSelected(null);
      setProgress(null);
      return;
    }

    let cancelled = false;
    async function loadArtifacts() {
      try {
        const [p, pcaRes, selRes] = await Promise.all([
          api.getExperimentProgress(selectedExperimentId!),
          api.getExperimentPca(selectedExperimentId!),
          api.getExperimentSelectedSamples(selectedExperimentId!),
        ]);
        if (!cancelled) {
          setProgress(p);
          setPca(pcaRes);
          setSelected(selRes);
        }
      } catch {
        // ignore transient fetch errors
      }
    }
    loadArtifacts();
    return () => {
      cancelled = true;
    };
  }, [selectedExperimentId, activeDetail?.summary.updated_at]);

  // Poll progress while running
  useEffect(() => {
    if (!selectedExperimentId) return;
    const isRunning =
      progress?.status === "running" ||
      activeDetail?.summary.status === "running";
    if (!isRunning) return;

    const timer = window.setInterval(async () => {
      try {
        const p = await api.getExperimentProgress(selectedExperimentId);
        setProgress(p);
        if (p.status !== "running") {
          onRefreshExperiment(selectedExperimentId);
          const [pcaRes, selRes] = await Promise.all([
            api.getExperimentPca(selectedExperimentId),
            api.getExperimentSelectedSamples(selectedExperimentId),
          ]);
          setPca(pcaRes);
          setSelected(selRes);
        }
      } catch {
        // ignore polling blip
      }
    }, 1000);

    return () => window.clearInterval(timer);
  }, [selectedExperimentId, progress?.status, activeDetail?.summary.status, onRefreshExperiment]);

  async function handleLaunch(e: React.FormEvent) {
    e.preventDefault();
    setError(null);
    setSubmitting(true);
    try {
      const payload: ExperimentCreateRequest = {
        name: name.trim() || undefined,
        dataset,
        model_type: modelType,
        strategy,
        mode,
        initial_size: initialSize,
        batch_size: batchSize,
        max_rounds: maxRounds,
        label_budget: labelBudget,
        max_samples: maxSamples,
        use_synthetic: useSynthetic,
        max_epochs: maxEpochs,
        lr,
        seed,
        label_noise_rate: noiseRate,
        warm_start: false,
        auto_start: true,
      };
      const created = await api.createExperiment(payload);
      onExperimentCreated(created.id);
      const p = await api.getExperimentProgress(created.id);
      setProgress(p);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to launch experiment.");
    } finally {
      setSubmitting(false);
    }
  }

  const rounds = activeDetail?.rounds || [];
  const activeSummary = activeDetail?.summary;
  const currentStatus = progress?.status || activeSummary?.status || "created";

  return (
    <div className="space-y-6">
      {error && <ErrorBanner message={error} />}

      {/* Configuration Panel */}
      <Card tone="secondary">
        <CardHeader>
          <div className="flex flex-col justify-between gap-2 sm:flex-row sm:items-center">
            <div>
              <CardTitle>Experiment Configuration & Execution</CardTitle>
              <CardDescription>
                Configure PyTorch training, query strategy, budget, and labeling mode
              </CardDescription>
            </div>
            {activeStrategyInfo && (
              <Badge variant="lavender" className="font-mono">
                {activeStrategyInfo.formula}
              </Badge>
            )}
          </div>
        </CardHeader>
        <CardContent>
          <form onSubmit={handleLaunch} className="space-y-4">
            <div className="space-y-3 rounded-xl border border-[#E6D6C6] bg-[#FAF3EC] p-4">
              <div className="text-[11px] font-semibold uppercase tracking-wider text-[#6B3E26]">
                1 · Domain, Model & Query Strategy
              </div>
              <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
                <div className="space-y-1.5">
                  <label className="text-xs font-medium text-[#806D60]">
                    Experiment Name (optional)
                  </label>
                  <Input
                    value={name}
                    onChange={(e) => setName(e.target.value)}
                    placeholder="e.g., CIFAR Hybrid Run"
                  />
                </div>

                <div className="space-y-1.5">
                  <label className="text-xs font-medium text-[#806D60]">
                    Dataset
                  </label>
                  <Select
                    value={dataset}
                    onChange={(e) => setDataset(e.target.value)}
                  >
                    {datasets.length > 0 ? (
                      datasets.map((d) => (
                        <option key={d.id} value={d.id}>
                          {d.name}
                        </option>
                      ))
                    ) : (
                      <>
                        <option value="cifar10">CIFAR-10 (Images)</option>
                        <option value="agnews">AG News (Text Embeddings)</option>
                      </>
                    )}
                  </Select>
                </div>

                <div className="space-y-1.5">
                  <label className="text-xs font-medium text-[#806D60]">
                    PyTorch Architecture
                  </label>
                  <Select
                    value={modelType}
                    onChange={(e) => setModelType(e.target.value)}
                  >
                    {compatibleModels.map((m) => (
                      <option key={m.id} value={m.id}>
                        {m.name}
                      </option>
                    ))}
                  </Select>
                </div>

                <div className="space-y-1.5">
                  <label className="text-xs font-medium text-[#806D60]">
                    Query Strategy
                  </label>
                  <Select
                    value={strategy}
                    onChange={(e) => setStrategy(e.target.value)}
                  >
                    {strategies.map((s) => (
                      <option key={s.id} value={s.id}>
                        {s.name} ({s.category})
                      </option>
                    ))}
                  </Select>
                </div>
              </div>
            </div>

            <div className="space-y-3 rounded-xl border border-[#E6D6C6] bg-[#FAF3EC] p-4">
              <div className="text-[11px] font-semibold uppercase tracking-wider text-[#6B3E26]">
                2 · Active Learning Budget & Loop Schedule
              </div>
              <div className="grid grid-cols-2 gap-4 sm:grid-cols-4 lg:grid-cols-7">
                <div className="col-span-2 space-y-1.5">
                  <label className="text-xs font-medium text-[#806D60]">
                    Labeling Mode
                  </label>
                  <Select
                    value={mode}
                    onChange={(e) => setMode(e.target.value as ExperimentMode)}
                  >
                    <option value="simulated_oracle">
                      Simulated Oracle (Auto-Run)
                    </option>
                    <option value="human_in_the_loop">
                      Human-in-the-Loop (Queue for Studio)
                    </option>
                  </Select>
                </div>

                <div className="space-y-1.5">
                  <label className="text-xs font-medium text-[#806D60]">
                    Initial Seed
                  </label>
                  <Input
                    type="number"
                    min={10}
                    max={200}
                    value={initialSize}
                    onChange={(e) => setInitialSize(Number(e.target.value))}
                  />
                </div>

                <div className="space-y-1.5">
                  <label className="text-xs font-medium text-[#806D60]">
                    Batch (K)
                  </label>
                  <Input
                    type="number"
                    min={1}
                    max={50}
                    value={batchSize}
                    onChange={(e) => setBatchSize(Number(e.target.value))}
                  />
                </div>

                <div className="space-y-1.5">
                  <label className="text-xs font-medium text-[#806D60]">
                    Rounds
                  </label>
                  <Input
                    type="number"
                    min={1}
                    max={20}
                    value={maxRounds}
                    onChange={(e) => setMaxRounds(Number(e.target.value))}
                  />
                </div>

                <div className="space-y-1.5">
                  <label className="text-xs font-medium text-[#806D60]">
                    Label Budget
                  </label>
                  <Input
                    type="number"
                    min={20}
                    max={5000}
                    value={labelBudget}
                    onChange={(e) => setLabelBudget(Number(e.target.value))}
                  />
                </div>

                <div className="space-y-1.5">
                  <label className="text-xs font-medium text-[#806D60]">
                    Pool Size
                  </label>
                  <Input
                    type="number"
                    min={80}
                    max={5000}
                    value={maxSamples}
                    onChange={(e) => setMaxSamples(Number(e.target.value))}
                  />
                </div>
              </div>
            </div>

            <div className="flex flex-wrap items-end justify-between gap-4 border-t border-[#D5DDF2] pt-4">
              <div className="flex flex-wrap items-center gap-4">
                <div className="w-24 space-y-1">
                  <label className="text-xs font-medium text-[#806D60]">Epochs</label>
                  <Input
                    type="number"
                    min={1}
                    max={50}
                    value={maxEpochs}
                    onChange={(e) => setMaxEpochs(Number(e.target.value))}
                  />
                </div>
                <div className="w-24 space-y-1">
                  <label className="text-xs font-medium text-[#806D60]">Seed</label>
                  <Input
                    type="number"
                    value={seed}
                    onChange={(e) => setSeed(Number(e.target.value))}
                  />
                </div>
                <div className="w-28 space-y-1">
                  <label className="text-xs font-medium text-[#806D60]">Oracle Noise</label>
                  <Input
                    type="number"
                    step="0.05"
                    min={0}
                    max={0.5}
                    value={noiseRate}
                    disabled={mode === "human_in_the_loop"}
                    onChange={(e) => setNoiseRate(Number(e.target.value))}
                  />
                </div>
                <label className="flex items-center gap-2 pt-4 text-xs text-[#38271F]">
                  <input
                    type="checkbox"
                    checked={useSynthetic}
                    disabled={dataset === "agnews"}
                    onChange={(e) => setUseSynthetic(e.target.checked)}
                    className="h-4 w-4 rounded border-[#E6D6C6] text-[#6B3E26]"
                  />
                  Fast Synthetic Mode (CPU)
                </label>
              </div>

              <div className="flex items-center gap-3">
                {currentStatus === "awaiting_annotations" && (
                  <Button
                    type="button"
                    variant="teal"
                    onClick={() => onNavigate("annotation")}
                  >
                    Open Annotation Studio ({progress?.n_queued ?? activeSummary?.n_queued ?? 0} Pending)
                  </Button>
                )}
                <Button type="submit" disabled={submitting}>
                  <Play className="h-4 w-4" />
                  {submitting
                    ? "Starting..."
                    : mode === "human_in_the_loop"
                      ? "Initialize & Queue Batch"
                      : "Execute Active Learning Loop"}
                </Button>
              </div>
            </div>
          </form>
        </CardContent>
      </Card>

      {/* Live Pipeline & Progress */}
      {activeSummary && (
        <div className="space-y-3 rounded-xl border border-[#BFDBFE] bg-[#FAF3EC] p-4">
          <div className="flex flex-wrap items-center justify-between gap-2">
            <div className="flex items-center gap-3">
              <h2 className="text-sm font-semibold text-[#38271F]">
                {activeSummary.name}
              </h2>
              <StatusBadge status={currentStatus} />
              <span className="font-mono text-xs text-[#806D60]">
                Round {progress?.current_round ?? activeSummary.current_round} /{" "}
                {activeSummary.max_rounds}
              </span>
            </div>
            <span className="font-mono text-xs font-medium text-[#6B3E26]">
              {progress?.current_stage || activeSummary.current_stage}
            </span>
          </div>
          <PipelineTracker
            stages={progress?.stages}
            activeIndex={progress?.stage_index ?? 0}
            isRunning={currentStatus === "running"}
            isCompleted={currentStatus === "completed"}
          />
          {progress?.error_message && (
            <ErrorBanner message={progress.error_message} />
          )}
        </div>
      )}

      {/* Large PCA Embedding Space + Learning Curve */}
      <div className="grid grid-cols-1 gap-6 xl:grid-cols-12">
        <Card className="xl:col-span-7">
          <CardHeader className="flex flex-row items-center justify-between gap-2">
            <div>
              <CardTitle>Latent Embedding Space (2D PCA Projection)</CardTitle>
              <CardDescription>
                Interactive Plotly projection of model bottleneck embeddings (Unlabeled vs Labeled vs Queried)
              </CardDescription>
            </div>
            {pca?.points.length ? (
              <Badge variant="default" className="font-mono">
                {pca.points.length} Embeddings
              </Badge>
            ) : null}
          </CardHeader>
          <CardContent>
            <PcaScatterPlot
              points={pca?.points || []}
              explainedVariance={pca?.explained_variance_ratio}
              height={440}
            />
          </CardContent>
        </Card>

        <Card tone="secondary" className="xl:col-span-5">
          <CardHeader>
            <CardTitle>Holdout Accuracy & Macro-F1</CardTitle>
            <CardDescription>
              Evaluated strictly on the locked 20% test split after Temperature Scaling
            </CardDescription>
          </CardHeader>
          <CardContent>
            <LearningCurveChart rounds={rounds} height={440} />
          </CardContent>
        </Card>
      </div>

      {/* Selected Samples Uncertainty & Diversity Diagnostics */}
      <Card tone="default">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Sparkles className="h-4 w-4 text-[#6B3E26]" />
            <span>Selected Batch: Uncertainty & Class Probability Analysis</span>
          </CardTitle>
          <CardDescription>
            {selected?.samples.length
              ? `Round ${selected.round_num} · Strategy: ${selected.strategy} · ${selected.samples.length} queried samples`
              : "Execute an active learning round to inspect why specific samples were queried."}
          </CardDescription>
        </CardHeader>
        <CardContent>
          {selected?.samples.length ? (
            <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
              {selected.samples.slice(0, 6).map((s) => (
                <div
                  key={s.sample_id}
                  className="grid grid-cols-1 gap-4 rounded-xl border border-[#E6D6C6] bg-[#FFFFFF] p-4 sm:grid-cols-12"
                >
                  <div className="space-y-2 sm:col-span-5">
                    <div className="font-mono text-xs font-semibold text-[#38271F]">
                      {s.sample_id}
                    </div>
                    {s.image_base64 && (
                      <div className="flex h-24 w-24 items-center justify-center rounded-lg border border-[#E6D6C6] bg-[#FAF3EC]">
                        <img
                          src={s.image_base64}
                          alt={s.sample_id}
                          className="h-20 w-20 object-contain"
                          style={{ imageRendering: "pixelated" }}
                        />
                      </div>
                    )}
                    <div className="space-y-1 text-xs">
                      <div className="flex justify-between">
                        <span className="text-[#806D60]">Model Guess:</span>
                        <span className="font-medium text-[#6B3E26]">
                          {s.predicted_class_name}
                        </span>
                      </div>
                      {s.true_class_name && (
                        <div className="flex justify-between">
                          <span className="text-[#806D60]">Oracle Truth:</span>
                          <span className="font-medium text-[#B98255]">
                            {s.true_class_name}
                          </span>
                        </div>
                      )}
                      <div className="flex justify-between">
                        <span className="text-[#806D60]">Confidence:</span>
                        <span className="font-mono text-[#38271F]">{formatPct(s.confidence)}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-[#806D60]">Uncertainty:</span>
                        <span className="font-mono text-[#B98255]">
                          {formatNum(s.uncertainty)}
                        </span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-[#806D60]">Entropy:</span>
                        <span className="font-mono text-[#38271F]">{formatNum(s.entropy)} bits</span>
                      </div>
                    </div>
                  </div>
                  <div className="sm:col-span-7">
                    <ClassProbabilityChart
                      classNames={s.class_names}
                      probabilities={s.probabilities}
                      predictedClassName={s.predicted_class_name}
                      height={190}
                    />
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <div className="py-8 text-center text-xs text-[#806D60]">
              No queried sample diagnostics available for this run yet.
            </div>
          )}
        </CardContent>
      </Card>

      {/* Round Telemetry + Execution Logs */}
      <div className="grid grid-cols-1 gap-6 lg:grid-cols-3">
        <Card tone="mint" className="lg:col-span-2">
          <CardHeader>
            <CardTitle>Round-by-Round Metrics</CardTitle>
          </CardHeader>
          <CardContent>
            {rounds.length ? (
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>Round</TableHead>
                    <TableHead>Labeled</TableHead>
                    <TableHead>Pool</TableHead>
                    <TableHead>Accuracy</TableHead>
                    <TableHead>Macro-F1</TableHead>
                    <TableHead>ECE</TableHead>
                    <TableHead>Temp (T)</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {rounds.map((r) => (
                    <TableRow key={r.round_num}>
                      <TableCell className="font-mono font-medium">
                        R{r.round_num}
                      </TableCell>
                      <TableCell className="font-mono">{r.n_labeled}</TableCell>
                      <TableCell className="font-mono">{r.n_pool}</TableCell>
                      <TableCell className="font-mono font-semibold text-[#6B3E26]">
                        {formatPct(r.test_acc)}
                      </TableCell>
                      <TableCell className="font-mono text-[#159C91]">
                        {formatPct(r.test_f1)}
                      </TableCell>
                      <TableCell className="font-mono">{formatNum(r.ece)}</TableCell>
                      <TableCell className="font-mono">
                        {formatNum(r.temperature, 2)}
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            ) : (
              <div className="py-6 text-center text-xs text-[#806D60]">
                Waiting for Round 0 evaluation...
              </div>
            )}
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Terminal className="h-4 w-4 text-[#6B3E26]" />
              <span>Engine Logs</span>
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="h-56 overflow-y-auto rounded-xl border border-[#E6D6C6] bg-[#1E293B] p-3 font-mono text-[11px] leading-relaxed text-[#93C5FD]">
              {(progress?.logs || activeDetail?.logs || ["No logs yet."]).map(
                (line, i) => (
                  <div key={i}>{line}</div>
                )
              )}
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}

