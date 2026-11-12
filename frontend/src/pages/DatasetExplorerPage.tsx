import { useEffect, useState } from "react";
import { Database, Layers, ShieldCheck, UploadCloud } from "lucide-react";
import { ClassDistributionChart } from "@/charts/ClassDistributionChart";
import {
  ErrorBanner,
  LoadingState,
  PendingFeatureBanner,
} from "@/components/shared/FeedbackBanner";
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
import { Select } from "@/components/ui/input";
import { api } from "@/services/api";
import type {
  DatasetSamplesResponse,
  DatasetStatsResponse,
  DatasetSummary,
  ExperimentSummary,
} from "@/types/api";

interface DatasetExplorerPageProps {
  datasets: DatasetSummary[];
  experiments: ExperimentSummary[];
  selectedExperimentId: string | null;
}

export function DatasetExplorerPage({
  datasets,
  experiments,
  selectedExperimentId,
}: DatasetExplorerPageProps) {
  const [datasetId, setDatasetId] = useState<string>("cifar10");
  const [useSynthetic, setUseSynthetic] = useState<boolean>(true);
  const [maxSamples, setMaxSamples] = useState<number>(450);
  const [linkedExpId, setLinkedExpId] = useState<string>(
    selectedExperimentId || ""
  );
  const [splitFilter, setSplitFilter] = useState<string>("");
  const [offset, setOffset] = useState<number>(0);

  const [stats, setStats] = useState<DatasetStatsResponse | null>(null);
  const [samplesRes, setSamplesRes] = useState<DatasetSamplesResponse | null>(
    null
  );
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  const activeMeta = datasets.find((d) => d.id === datasetId);

  useEffect(() => {
    let cancelled = false;
    async function load() {
      setLoading(true);
      setError(null);
      try {
        const expMatch =
          linkedExpId &&
          experiments.find((e) => e.id === linkedExpId)?.dataset === datasetId
            ? linkedExpId
            : undefined;

        const [s, samp] = await Promise.all([
          api.getDatasetStats(datasetId, {
            max_samples: maxSamples,
            use_synthetic: useSynthetic,
            experiment_id: expMatch,
          }),
          api.getDatasetSamples(datasetId, {
            offset,
            limit: 12,
            max_samples: maxSamples,
            use_synthetic: useSynthetic,
            split: splitFilter || undefined,
            experiment_id: expMatch,
          }),
        ]);
        if (!cancelled) {
          setStats(s);
          setSamplesRes(samp);
        }
      } catch (err) {
        if (!cancelled) {
          setError(err instanceof Error ? err.message : "Failed to load dataset.");
        }
      } finally {
        if (!cancelled) setLoading(false);
      }
    }
    load();
    return () => {
      cancelled = true;
    };
  }, [datasetId, useSynthetic, maxSamples, linkedExpId, splitFilter, offset, experiments]);

  return (
    <div className="space-y-6">
      {/* Filter Controls */}
      <Card>
        <CardContent className="grid grid-cols-1 gap-4 p-5 sm:grid-cols-2 lg:grid-cols-5">
          <div className="space-y-1.5">
            <label className="text-xs font-medium text-[#64748B]">
              Dataset Domain
            </label>
            <Select
              value={datasetId}
              onChange={(e) => {
                setDatasetId(e.target.value);
                setOffset(0);
              }}
            >
              <option value="cifar10">CIFAR-10 (Image Vision)</option>
              <option value="agnews">AG News (Text Embeddings)</option>
            </Select>
          </div>

          <div className="space-y-1.5">
            <label className="text-xs font-medium text-[#64748B]">
              Data Source Mode
            </label>
            <Select
              value={useSynthetic ? "synthetic" : "real"}
              onChange={(e) => {
                setUseSynthetic(e.target.value === "synthetic");
                setOffset(0);
              }}
              disabled={datasetId === "agnews"}
            >
              <option value="synthetic">Synthetic Fast-CPU Subset</option>
              <option value="real">Real Torchvision Download</option>
            </Select>
          </div>

          <div className="space-y-1.5">
            <label className="text-xs font-medium text-[#64748B]">
              Pool Sample Cap
            </label>
            <Select
              value={String(maxSamples)}
              onChange={(e) => {
                setMaxSamples(Number(e.target.value));
                setOffset(0);
              }}
            >
              <option value="200">200 samples</option>
              <option value="450">450 samples</option>
              <option value="800">800 samples</option>
              <option value="1200">1,200 samples</option>
            </Select>
          </div>

          <div className="space-y-1.5">
            <label className="text-xs font-medium text-[#64748B]">
              Split Filter
            </label>
            <Select
              value={splitFilter}
              onChange={(e) => {
                setSplitFilter(e.target.value);
                setOffset(0);
              }}
            >
              <option value="">All Splits (Pool / Val / Test)</option>
              <option value="pool">Unlabeled/Labeled Pool (70%)</option>
              <option value="val">Validation Calibration (10%)</option>
              <option value="test">Holdout Test Set (20%)</option>
            </Select>
          </div>

          <div className="space-y-1.5">
            <label className="text-xs font-medium text-[#64748B]">
              Inspect Experiment DB
            </label>
            <Select
              value={linkedExpId}
              onChange={(e) => {
                setLinkedExpId(e.target.value);
                setOffset(0);
              }}
            >
              <option value="">Deterministic Split Preview</option>
              {experiments
                .filter((e) => e.dataset === datasetId)
                .map((e) => (
                  <option key={e.id} value={e.id}>
                    {e.name}
                  </option>
                ))}
            </Select>
          </div>
        </CardContent>
      </Card>

      {error && <ErrorBanner message={error} />}

      {loading && !stats ? (
        <LoadingState label="Loading dataset statistics and sample previews..." />
      ) : (
        stats && (
          <>
            {/* Overview KPI Cards */}
            <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 xl:grid-cols-4">
              <MetricCard
                title="Total Loaded Samples"
                value={String(stats.split_stats.n_total)}
                subtitle={`Tensor shape: [${stats.feature_shape.join(" × ")}]`}
                icon={Database}
                accent="blue"
              />
              <MetricCard
                title="Unlabeled Pool"
                value={String(stats.split_stats.n_pool_unlabeled)}
                subtitle={`${stats.split_stats.n_labeled} labeled · ${stats.split_stats.n_queued} queued`}
                icon={Layers}
                accent="teal"
              />
              <MetricCard
                title="Zero-Leakage Holdout Test"
                value={String(stats.split_stats.n_test)}
                subtitle={`${stats.split_stats.n_val} validation calibration samples`}
                icon={ShieldCheck}
                accent="green"
              />
              <MetricCard
                title="Target Classes"
                value={String(stats.num_classes)}
                subtitle={stats.modality === "image" ? "RGB 32×32 Images" : "64-d Text Embeddings"}
                icon={Database}
                accent="slate"
              />
            </div>

            {/* Class Distribution + Metadata */}
            <div className="grid grid-cols-1 gap-6 lg:grid-cols-3">
              <Card className="lg:col-span-2">
                <CardHeader>
                  <CardTitle>Class Distribution</CardTitle>
                  <CardDescription>
                    Ground-truth class balance across the {stats.max_samples} loaded samples
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <ClassDistributionChart data={stats.class_distribution} height={280} />
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle>Dataset Adapter Metadata</CardTitle>
                  <CardDescription>
                    Backend adapter capabilities in src/data/
                  </CardDescription>
                </CardHeader>
                <CardContent className="space-y-3 text-xs">
                  <div className="flex justify-between border-b border-[#E2E8F0] pb-2">
                    <span className="text-[#64748B]">Adapter ID</span>
                    <span className="font-mono font-medium">{stats.dataset_id}</span>
                  </div>
                  <div className="flex justify-between border-b border-[#E2E8F0] pb-2">
                    <span className="text-[#64748B]">Modality</span>
                    <Badge variant="default">{stats.modality}</Badge>
                  </div>
                  <div className="flex justify-between border-b border-[#E2E8F0] pb-2">
                    <span className="text-[#64748B]">Supported Models</span>
                    <span className="font-mono">
                      {activeMeta?.supported_models.join(", ") || "—"}
                    </span>
                  </div>
                  <div className="flex justify-between border-b border-[#E2E8F0] pb-2">
                    <span className="text-[#64748B]">Raw Text Storage</span>
                    <span className="font-mono">
                      {activeMeta?.supports_raw_text ? "Yes" : "64-d Embeddings Only"}
                    </span>
                  </div>
                  <p className="text-[#64748B]">{activeMeta?.description}</p>

                  <div className="rounded-lg border border-dashed border-[#CBD5E1] bg-[#F8FAFC] p-3">
                    <div className="flex items-center gap-2 font-medium text-[#172033]">
                      <UploadCloud className="h-4 w-4 text-[#64748B]" />
                      <span>Custom Dataset Upload</span>
                      <Badge variant="warning">Pending</Badge>
                    </div>
                    <p className="mt-1 text-[11px] text-[#64748B]">
                      {stats.upload_status_note}
                    </p>
                  </div>
                </CardContent>
              </Card>
            </div>

            {/* Sample Preview Gallery */}
            <Card>
              <CardHeader className="flex flex-row items-center justify-between">
                <div>
                  <CardTitle>Sample Preview</CardTitle>
                  <CardDescription>
                    Showing {samplesRes?.samples.length || 0} of {samplesRes?.total || 0} samples
                  </CardDescription>
                </div>
                <div className="flex items-center gap-2">
                  <Button
                    variant="outline"
                    size="sm"
                    disabled={offset === 0}
                    onClick={() => setOffset((v) => Math.max(0, v - 12))}
                  >
                    Previous
                  </Button>
                  <Button
                    variant="outline"
                    size="sm"
                    disabled={!samplesRes || offset + 12 >= samplesRes.total}
                    onClick={() => setOffset((v) => v + 12)}
                  >
                    Next
                  </Button>
                </div>
              </CardHeader>
              <CardContent>
                {datasetId === "agnews" && (
                  <PendingFeatureBanner
                    className="mb-4"
                    title="AG News Representation"
                    description="The existing AGNewsAdapter produces 64-dimensional synthetic dense embeddings rather than raw article text strings, so embedding statistics and top active dimensions are displayed below."
                  />
                )}
                <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 md:grid-cols-3 xl:grid-cols-4">
                  {samplesRes?.samples.map((s) => (
                    <div
                      key={s.sample_id}
                      className="rounded-lg border border-[#E2E8F0] bg-[#F8FAFC] p-3"
                    >
                      <div className="mb-2 flex items-center justify-between gap-2">
                        <span className="truncate font-mono text-[11px] font-medium text-[#172033]">
                          {s.sample_id}
                        </span>
                        <Badge
                          variant={
                            s.split === "test"
                              ? "warning"
                              : s.status === "labeled"
                                ? "success"
                                : "neutral"
                          }
                        >
                          {s.split}:{s.status}
                        </Badge>
                      </div>

                      {s.image_base64 ? (
                        <div className="mb-2.5 flex h-28 items-center justify-center rounded-md border border-[#E2E8F0] bg-white">
                          <img
                            src={s.image_base64}
                            alt={s.sample_id}
                            className="h-24 w-24 rounded object-contain"
                            style={{ imageRendering: "pixelated" }}
                          />
                        </div>
                      ) : (
                        s.feature_summary && (
                          <div className="mb-2.5 rounded-md border border-[#E2E8F0] bg-white p-2 font-mono text-[11px] text-[#475569]">
                            <div>L2 Norm: {s.feature_summary.l2_norm}</div>
                            <div>
                              Mean/Std: {s.feature_summary.mean} / {s.feature_summary.std}
                            </div>
                            <div className="truncate text-[#64748B]">
                              Vec: [{s.embedding_preview?.slice(0, 5).join(", ")}...]
                            </div>
                          </div>
                        )
                      )}

                      <div className="flex items-center justify-between text-xs">
                        <span className="text-[#64748B]">True Class:</span>
                        <span className="font-medium text-[#172033]">
                          {s.true_class_name || "—"}
                        </span>
                      </div>
                      {s.annotator && (
                        <div className="mt-1 flex items-center justify-between text-[11px] text-[#64748B]">
                          <span>Source:</span>
                          <span className="font-mono">{s.annotator}</span>
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          </>
        )
      )}
    </div>
  );
}

