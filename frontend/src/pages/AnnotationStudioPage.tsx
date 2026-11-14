import { useCallback, useEffect, useState } from "react";
import {
  CheckCircle2,
  ChevronLeft,
  ChevronRight,
  EyeOff,
  Play,
  RefreshCw,
  Webhook,
} from "lucide-react";
import { ClassProbabilityChart } from "@/charts/ClassProbabilityChart";
import {
  ErrorBanner,
  LoadingState,
  PendingFeatureBanner,
} from "@/components/shared/FeedbackBanner";
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
  AnnotationHistoryResponse,
  ExperimentSummary,
  LabelStudioStatusResponse,
  PendingAnnotationsResponse,
} from "@/types/api";
import { formatNum, formatPct, formatShortDate } from "@/utils/format";

interface AnnotationStudioPageProps {
  experiments: ExperimentSummary[];
  selectedExperimentId: string | null;
  onSelectExperiment: (id: string) => void;
  onExperimentCreated: (id: string) => void;
  onRefreshExperiment: (id: string) => void;
}

export function AnnotationStudioPage({
  experiments,
  selectedExperimentId,
  onSelectExperiment,
  onExperimentCreated,
  onRefreshExperiment,
}: AnnotationStudioPageProps) {
  const hitlExperiments = experiments.filter(
    (e) => e.mode === "human_in_the_loop"
  );
  const currentExp =
    experiments.find((e) => e.id === selectedExperimentId) ||
    hitlExperiments[0] ||
    experiments[0] ||
    null;

  const [annotatorName, setAnnotatorName] = useState("human_reviewer");
  const [pending, setPending] = useState<PendingAnnotationsResponse | null>(null);
  const [history, setHistory] = useState<AnnotationHistoryResponse | null>(null);
  const [lsStatus, setLsStatus] = useState<LabelStudioStatusResponse | null>(null);
  const [sampleIndex, setSampleIndex] = useState(0);
  const [selectedLabelId, setSelectedLabelId] = useState<number | null>(null);

  const [loading, setLoading] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [advancing, setAdvancing] = useState(false);
  const [bootstrapping, setBootstrapping] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const loadData = useCallback(async (expId: string) => {
    setLoading(true);
    setError(null);
    try {
      const [pendRes, histRes, lsRes] = await Promise.all([
        api.getPendingAnnotations(expId),
        api.getAnnotationHistory(expId),
        api.getLabelStudioStatus(),
      ]);
      setPending(pendRes);
      setHistory(histRes);
      setLsStatus(lsRes);
      setSampleIndex((idx) =>
        pendRes.samples.length === 0
          ? 0
          : Math.min(idx, pendRes.samples.length - 1)
      );
      setSelectedLabelId(null);
    } catch (err) {
      setError(
        err instanceof Error ? err.message : "Failed to load annotation queue."
      );
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    if (currentExp) {
      loadData(currentExp.id);
    } else {
      api.getLabelStudioStatus().then(setLsStatus).catch(() => {});
    }
  }, [currentExp?.id, currentExp?.status, loadData]);

  async function handleQuickStartHitl(dataset: "cifar10" | "agnews") {
    setBootstrapping(true);
    setError(null);
    try {
      const created = await api.createExperiment({
        name: `HITL ${dataset.toUpperCase()} Session`,
        dataset,
        model_type: dataset === "cifar10" ? "cifar_cnn" : "text_mlp",
        strategy: "hybrid",
        mode: "human_in_the_loop",
        initial_size: 20,
        batch_size: 6,
        max_rounds: 4,
        max_samples: 300,
        use_synthetic: true,
        max_epochs: 5,
        lr: dataset === "cifar10" ? 0.01 : 0.02,
        seed: 42,
        label_noise_rate: 0.0,
        warm_start: false,
        auto_start: true,
      });
      onExperimentCreated(created.id);
      // Poll briefly until initial batch is queued
      for (let i = 0; i < 40; i++) {
        await new Promise((r) => setTimeout(r, 500));
        const p = await api.getExperimentProgress(created.id);
        if (p.status !== "running") {
          onRefreshExperiment(created.id);
          await loadData(created.id);
          break;
        }
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to start HITL session.");
    } finally {
      setBootstrapping(false);
    }
  }

  async function handleSubmitLabel() {
    if (!currentExp || !activeSample || selectedLabelId === null) return;
    setSubmitting(true);
    setError(null);
    try {
      await api.submitAnnotation({
        experiment_id: currentExp.id,
        sample_id: activeSample.sample_id,
        label_id: selectedLabelId,
        annotator: annotatorName.trim() || "human_ui",
      });
      await loadData(currentExp.id);
      onRefreshExperiment(currentExp.id);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to submit label.");
    } finally {
      setSubmitting(false);
    }
  }

  async function handleRetrainAndAdvance() {
    if (!currentExp) return;
    setAdvancing(true);
    setError(null);
    try {
      await api.advanceExperiment(currentExp.id);
      for (let i = 0; i < 50; i++) {
        await new Promise((r) => setTimeout(r, 600));
        const p = await api.getExperimentProgress(currentExp.id);
        if (p.status !== "running") {
          onRefreshExperiment(currentExp.id);
          await loadData(currentExp.id);
          break;
        }
      }
    } catch (err) {
      setError(
        err instanceof Error ? err.message : "Failed to advance round."
      );
    } finally {
      setAdvancing(false);
    }
  }

  const samples = pending?.samples || [];
  const activeSample = samples[sampleIndex] || null;
  const totalBatch =
    (pending?.pending_count || 0) + (pending?.completed_human_count || 0);
  const progressPct =
    totalBatch > 0
      ? ((pending?.completed_human_count || 0) / totalBatch) * 100
      : 0;

  return (
    <div className="space-y-6">
      {/* Top Selector & Quick HITL Starter */}
      <Card tone="secondary">
        <CardContent className="flex flex-col justify-between gap-4 p-5 lg:flex-row lg:items-center">
          <div className="flex flex-wrap items-center gap-3">
            <div className="w-72">
              <label className="mb-1 block text-xs font-medium text-[#64748B]">
                Workspace Experiment
              </label>
              <Select
                value={currentExp?.id || ""}
                onChange={(e) => onSelectExperiment(e.target.value)}
              >
                {experiments.length === 0 && (
                  <option value="">No experiments created yet</option>
                )}
                {experiments.map((exp) => (
                  <option key={exp.id} value={exp.id}>
                    {exp.name} ({exp.mode === "human_in_the_loop" ? "HITL" : "Oracle"})
                  </option>
                ))}
              </Select>
            </div>

            <div className="w-48">
              <label className="mb-1 block text-xs font-medium text-[#64748B]">
                Annotator ID
              </label>
              <Input
                value={annotatorName}
                onChange={(e) => setAnnotatorName(e.target.value)}
                placeholder="human_reviewer"
              />
            </div>

            {currentExp && (
              <div className="pt-5">
                <StatusBadge status={currentExp.status} />
              </div>
            )}
          </div>

          <div className="flex flex-wrap items-center gap-2">
            <Button
              variant="outline"
              size="sm"
              disabled={bootstrapping}
              onClick={() => handleQuickStartHitl("cifar10")}
            >
              <Play className="h-3.5 w-3.5" />
              {bootstrapping ? "Preparing Queue..." : "New HITL Image Batch"}
            </Button>
            <Button
              variant="outline"
              size="sm"
              disabled={bootstrapping}
              onClick={() => handleQuickStartHitl("agnews")}
            >
              <Play className="h-3.5 w-3.5" />
              New HITL Text Batch
            </Button>
          </div>
        </CardContent>
      </Card>

      {error && <ErrorBanner message={error} />}

      {/* Blind Annotation Guarantee Banner */}
      <div className="flex flex-wrap items-center justify-between gap-2 rounded-2xl border border-[#AEBEFB] bg-[#DCE5FF]/70 px-4 py-3 text-xs text-[#263452]">
        <div className="flex items-center gap-2">
          <EyeOff className="h-4 w-4 shrink-0 text-[#4568D9]" />
          <span>
            <strong>Blind Annotation Mode:</strong> Ground-truth labels are strictly excluded from the pending annotation API response until after submission.
          </span>
        </div>
        {currentExp?.mode === "simulated_oracle" && (
          <Badge variant="warning">
            Viewing Simulated Oracle Run (switch or start HITL batch to annotate manually)
          </Badge>
        )}
      </div>

      {/* Progress & Counters */}
      {pending && (
        <Card tone="blue">
          <CardContent className="grid grid-cols-1 items-center gap-4 p-5 md:grid-cols-4">
            <div>
              <p className="text-xs font-medium text-[#64748B]">Pending in Queue</p>
              <p className="font-mono text-2xl font-semibold text-[#4568D9]">
                {pending.pending_count}
              </p>
            </div>
            <div>
              <p className="text-xs font-medium text-[#64748B]">Human Labels Submitted</p>
              <p className="font-mono text-2xl font-semibold text-[#159C91]">
                {pending.completed_human_count}
              </p>
            </div>
            <div>
              <p className="text-xs font-medium text-[#64748B]">Total Labeled (Incl. Seed)</p>
              <p className="font-mono text-2xl font-semibold text-[#263452]">
                {pending.total_labeled_count}
              </p>
            </div>
            <div className="space-y-2">
              <div className="flex justify-between text-xs">
                <span className="font-medium text-[#64748B]">Human Batch Progress</span>
                <span className="font-mono font-semibold text-[#263452]">
                  {progressPct.toFixed(0)}%
                </span>
              </div>
              <Progress
                value={progressPct}
                indicatorClassName="bg-[#159C91]"
              />
            </div>
          </CardContent>
        </Card>
      )}

      {/* Main Annotation Cockpit */}
      {loading && !pending ? (
        <LoadingState label="Loading queued samples..." />
      ) : activeSample ? (
        <div className="grid grid-cols-1 gap-6 lg:grid-cols-12">
          {/* Left: Primary Focus Sample Viewer */}
          <Card tone="default" className="flex flex-col lg:col-span-6">
            <CardHeader className="flex flex-row items-center justify-between border-b border-[#D5DDF2] pb-4">
              <div>
                <div className="flex items-center gap-2">
                  <CardTitle className="font-mono text-sm">
                    {activeSample.sample_id}
                  </CardTitle>
                  <Badge variant="default">Round {activeSample.round_num}</Badge>
                </div>
                <CardDescription className="mt-1">
                  Sample <span className="font-mono font-semibold text-[#263452]">{sampleIndex + 1}</span> of{" "}
                  <span className="font-mono">{samples.length}</span> queued for review
                </CardDescription>
              </div>
              <div className="flex items-center gap-1.5">
                <Button
                  variant="outline"
                  size="sm"
                  disabled={sampleIndex === 0}
                  onClick={() => {
                    setSampleIndex((i) => Math.max(0, i - 1));
                    setSelectedLabelId(null);
                  }}
                >
                  <ChevronLeft className="h-4 w-4" />
                  Prev
                </Button>
                <Button
                  variant="outline"
                  size="sm"
                  disabled={sampleIndex >= samples.length - 1}
                  onClick={() => {
                    setSampleIndex((i) => Math.min(samples.length - 1, i + 1));
                    setSelectedLabelId(null);
                  }}
                >
                  Next
                  <ChevronRight className="h-4 w-4" />
                </Button>
              </div>
            </CardHeader>
            <CardContent className="flex flex-1 flex-col justify-center p-6">
              {activeSample.modality === "image" && activeSample.image_base64 ? (
                <div className="flex h-80 items-center justify-center rounded-2xl border border-[#C7D3F0] bg-[#E9EDFA] p-6">
                  <img
                    src={activeSample.image_base64}
                    alt={activeSample.sample_id}
                    className="h-60 w-60 rounded-xl object-contain shadow-sm"
                    style={{ imageRendering: "pixelated" }}
                  />
                </div>
              ) : (
                <div className="space-y-4 rounded-2xl border border-[#C7D3F0] bg-[#E9EDFA] p-6">
                  <div className="flex items-center justify-between">
                    <span className="text-sm font-semibold text-[#263452]">
                      AG News 64-d Text Embedding Vector
                    </span>
                    <Badge variant="lavender">Embedding Representation</Badge>
                  </div>
                  <p className="text-xs leading-relaxed text-[#64748B]">
                    Current backend adapter represents news items as 64-dimensional normalized feature vectors rather than raw text strings.
                  </p>
                  {activeSample.feature_summary && (
                    <div className="grid grid-cols-2 gap-3 rounded-xl border border-[#D5DDF2] bg-[#F3F5FD] p-4 font-mono text-xs">
                      {Object.entries(activeSample.feature_summary).map(
                        ([k, v]) => (
                          <div key={k} className="flex justify-between">
                            <span className="text-[#64748B]">{k}:</span>
                            <span className="font-semibold text-[#263452]">{v}</span>
                          </div>
                        )
                      )}
                    </div>
                  )}
                </div>
              )}
            </CardContent>
          </Card>

          {/* Right Column: Separated Predictions/Confidence Card & Annotation Controls Card */}
          <div className="space-y-6 lg:col-span-6">
            {/* 1. Model Predictions & Confidence Card */}
            <Card tone="lavender">
              <CardHeader className="pb-3">
                <CardTitle>Model Prediction & Uncertainty</CardTitle>
                <CardDescription>
                  Auxiliary calibrated signals (ground truth remains hidden prior to submission)
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="grid grid-cols-3 gap-3 rounded-xl border border-[#D4C3F5] bg-[#F3F5FD]/90 p-3 text-center">
                  <div>
                    <div className="text-[11px] font-medium text-[#64748B]">
                      Model Guess
                    </div>
                    <div className="mt-0.5 text-xs font-semibold text-[#4568D9]">
                      {activeSample.predicted_class_name || "—"}
                    </div>
                  </div>
                  <div>
                    <div className="text-[11px] font-medium text-[#64748B]">
                      Confidence
                    </div>
                    <div className="mt-0.5 font-mono text-xs font-semibold text-[#263452]">
                      {formatPct(activeSample.confidence)}
                    </div>
                  </div>
                  <div>
                    <div className="text-[11px] font-medium text-[#64748B]">
                      Uncertainty
                    </div>
                    <div className="mt-0.5 font-mono text-xs font-semibold text-[#B45309]">
                      {formatNum(activeSample.uncertainty)}
                    </div>
                  </div>
                </div>

                {activeSample.probabilities && (
                  <div>
                    <p className="mb-1.5 text-xs font-medium text-[#64748B]">
                      Calibrated Class Probabilities
                    </p>
                    <ClassProbabilityChart
                      classNames={activeSample.class_names}
                      probabilities={activeSample.probabilities}
                      predictedClassName={activeSample.predicted_class_name}
                      height={195}
                    />
                  </div>
                )}
              </CardContent>
            </Card>

            {/* 2. Human Annotation Controls Card */}
            <Card tone="mint">
              <CardHeader className="pb-3">
                <CardTitle>Annotation Controls</CardTitle>
                <CardDescription>
                  Select the verified category label to record into SQLite
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="grid grid-cols-2 gap-2.5 sm:grid-cols-3 md:grid-cols-5">
                  {activeSample.class_names.map((clsName, idx) => {
                    const isSelected = selectedLabelId === idx;
                    const isPred = activeSample.predicted_class_id === idx;
                    return (
                      <button
                        key={clsName}
                        type="button"
                        onClick={() => setSelectedLabelId(idx)}
                        className={`flex flex-col items-start rounded-xl border px-3 py-2 text-left transition-all ${
                          isSelected
                            ? "border-[#4568D9] bg-[#DCE5FF] ring-2 ring-[#4568D9]/25"
                            : "border-[#BDE4DA] bg-[#F3F5FD] hover:border-[#718BE0]"
                        }`}
                      >
                        <span className="font-mono text-[10px] text-[#64748B]">
                          Class {idx}
                        </span>
                        <span className="text-xs font-semibold text-[#263452]">
                          {clsName}
                        </span>
                        {isPred && (
                          <span className="mt-1 text-[10px] font-medium text-[#4568D9]">
                            Suggested
                          </span>
                        )}
                      </button>
                    );
                  })}
                </div>

                <div className="flex flex-wrap items-center justify-between gap-3 border-t border-[#BDE4DA] pt-4">
                  <div className="text-xs text-[#64748B]">
                    {selectedLabelId !== null ? (
                      <span>
                        Selected Label:{" "}
                        <strong className="font-semibold text-[#263452]">
                          {activeSample.class_names[selectedLabelId]}
                        </strong>
                      </span>
                    ) : (
                      "Click a category above to annotate this sample."
                    )}
                  </div>
                  <Button
                    variant="teal"
                    disabled={selectedLabelId === null || submitting}
                    onClick={handleSubmitLabel}
                  >
                    <CheckCircle2 className="h-4 w-4" />
                    {submitting ? "Saving to SQLite..." : "Submit Annotation"}
                  </Button>
                </div>
              </CardContent>
            </Card>
          </div>
        </div>
      ) : (
        <Card tone="mint">
          <CardContent className="flex flex-col items-center justify-center gap-3 py-10 text-center">
            <CheckCircle2 className="h-8 w-8 text-[#159C91]" />
            <div>
              <h3 className="text-sm font-semibold text-[#263452]">
                {currentExp?.mode === "human_in_the_loop"
                  ? currentExp.status === "completed"
                    ? "Human-in-the-Loop Experiment Completed"
                    : "All Queued Samples Annotated!"
                  : "No Samples Queued for Human Annotation"}
              </h3>
              <p className="mt-1 text-xs text-[#64748B]">
                {currentExp?.mode === "human_in_the_loop" &&
                currentExp.status !== "completed"
                  ? "Ready to retrain the PyTorch model with your human labels and query the next uncertain batch."
                  : "Click 'New HITL Image Batch' above to launch an interactive Human-in-the-Loop round."}
              </p>
            </div>
            {currentExp?.mode === "human_in_the_loop" &&
              currentExp.status === "awaiting_annotations" &&
              pending?.pending_count === 0 && (
                <Button
                  variant="default"
                  disabled={advancing}
                  onClick={handleRetrainAndAdvance}
                >
                  <RefreshCw
                    className={`h-4 w-4 ${advancing ? "animate-spin" : ""}`}
                  />
                  {advancing
                    ? "Retraining & Querying Next Batch..."
                    : "Retrain Model & Advance Round"}
                </Button>
              )}
          </CardContent>
        </Card>
      )}

      {/* Label Studio Phase 3 Readiness + Annotation History */}
      <div className="grid grid-cols-1 gap-6 lg:grid-cols-3">
        <Card tone="secondary">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Webhook className="h-4 w-4 text-[#159C91]" />
              <span>Label Studio Integration</span>
            </CardTitle>
            <CardDescription>
              Phase 3 Human-in-the-Loop Readiness
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-3 text-xs">
            <div className="flex items-center justify-between">
              <span className="text-[#64748B]">Status</span>
              <Badge variant="warning">
                {lsStatus?.integration_status || "pending_phase_3"}
              </Badge>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-[#64748B]">Webhook Route</span>
              <code className="rounded-md bg-[#DCE5FF] px-2 py-0.5 font-mono text-[11px] text-[#263452]">
                {lsStatus?.webhook_endpoint || "/api/v1/webhooks/label-studio"}
              </code>
            </div>
            <PendingFeatureBanner
              title="Phase 3 Roadmap"
              description={
                lsStatus?.notes ||
                "External Label Studio SDK sync and Cohen's Kappa QA are pending Phase 3."
              }
            />
          </CardContent>
        </Card>

        <Card className="lg:col-span-2">
          <CardHeader className="flex flex-row items-center justify-between">
            <div>
              <CardTitle>Annotation Audit Log</CardTitle>
              <CardDescription>
                Explicitly separates Human UI annotations from Simulated Oracle & Cold-Start seeds
              </CardDescription>
            </div>
            {history && (
              <div className="flex gap-2">
                <Badge variant="teal">Human: {history.human_count}</Badge>
                <Badge variant="neutral">Oracle: {history.oracle_count}</Badge>
              </div>
            )}
          </CardHeader>
          <CardContent>
            {history?.items.length ? (
              <div className="max-h-72 overflow-y-auto">
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead>Sample ID</TableHead>
                      <TableHead>Assigned Label</TableHead>
                      <TableHead>Source</TableHead>
                      <TableHead>Round</TableHead>
                      <TableHead>Truth Check</TableHead>
                      <TableHead>Updated</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {history.items.slice(0, 30).map((item) => (
                      <TableRow key={item.sample_id}>
                        <TableCell className="font-mono text-xs">
                          {item.sample_id}
                        </TableCell>
                        <TableCell className="text-xs font-medium">
                          {item.assigned_label_name}
                        </TableCell>
                        <TableCell>
                          <Badge
                            variant={
                              item.source_type === "human"
                                ? "teal"
                                : item.source_type === "simulated_oracle"
                                  ? "warning"
                                  : "neutral"
                            }
                          >
                            {item.annotator}
                          </Badge>
                        </TableCell>
                        <TableCell className="font-mono text-xs">
                          R{item.round_added ?? 0}
                        </TableCell>
                        <TableCell className="text-xs">
                          {item.matches_ground_truth === null ? (
                            "—"
                          ) : item.matches_ground_truth ? (
                            <span className="font-medium text-[#16A34A]">
                              Match ({item.true_label_name})
                            </span>
                          ) : (
                            <span className="font-medium text-[#B45309]">
                              True: {item.true_label_name}
                            </span>
                          )}
                        </TableCell>
                        <TableCell className="text-xs text-[#64748B]">
                          {formatShortDate(item.updated_at)}
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </div>
            ) : (
              <div className="py-8 text-center text-xs text-[#64748B]">
                No labeled pool records yet for this experiment.
              </div>
            )}
          </CardContent>
        </Card>
      </div>
    </div>
  );
}

