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

  // Custom Dataset Upload state
  const [uploadModalOpen, setUploadModalOpen] = useState(false);
  const [uploadFile, setUploadFile] = useState<File | null>(null);
  const [uploadName, setUploadName] = useState("");
  const [uploadModality, setUploadModality] = useState<"image" | "text">("image");
  const [uploading, setUploading] = useState(false);
  const [uploadMsg, setUploadMsg] = useState<{ type: "success" | "error"; text: string } | null>(null);

  const activeMeta = datasets.find((d) => d.id === datasetId);

  const handleUploadSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!uploadFile || !uploadName.trim()) {
      setUploadMsg({ type: "error", text: "Please provide a dataset name and select a file." });
      return;
    }
    setUploading(true);
    setUploadMsg(null);
    try {
      const fd = new FormData();
      fd.append("file", uploadFile);
      fd.append("name", uploadName.trim());
      fd.append("modality", uploadModality);
      const res = await api.uploadDataset(fd);
      setUploadMsg({
        type: "success",
        text: `Successfully uploaded ${res.name}! Refreshing catalog...`,
      });
      setTimeout(() => {
        window.location.reload();
      }, 1200);
    } catch (err: any) {
      setUploadMsg({
        type: "error",
        text: err.message || "Upload failed. Please check archive format.",
      });
    } finally {
      setUploading(false);
    }
  };

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
      <Card tone="secondary">
        <CardContent className="grid grid-cols-1 gap-4 p-5 sm:grid-cols-2 lg:grid-cols-5">
          <div className="space-y-1.5">
            <label className="text-xs font-medium text-[#806D60]">
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
            <label className="text-xs font-medium text-[#806D60]">
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
            <label className="text-xs font-medium text-[#806D60]">
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
            <label className="text-xs font-medium text-[#806D60]">
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
            <label className="text-xs font-medium text-[#806D60]">
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
                tone="default"
              />
              <MetricCard
                title="Unlabeled Pool"
                value={String(stats.split_stats.n_pool_unlabeled)}
                subtitle={`${stats.split_stats.n_labeled} labeled · ${stats.split_stats.n_queued} queued`}
                icon={Layers}
                accent="teal"
                tone="mint"
              />
              <MetricCard
                title="Zero-Leakage Holdout Test"
                value={String(stats.split_stats.n_test)}
                subtitle={`${stats.split_stats.n_val} validation calibration samples`}
                icon={ShieldCheck}
                accent="green"
                tone="lavender"
              />
              <MetricCard
                title="Target Classes"
                value={String(stats.num_classes)}
                subtitle={stats.modality === "image" ? "RGB 32×32 Images" : "64-d Text Embeddings"}
                icon={Database}
                accent="slate"
                tone="secondary"
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
                  <ClassDistributionChart data={stats.class_distribution} height={310} />
                </CardContent>
              </Card>

              <Card tone="lavender">
                <CardHeader>
                  <CardTitle>Dataset Adapter Metadata</CardTitle>
                  <CardDescription>
                    Backend adapter capabilities in src/data/
                  </CardDescription>
                </CardHeader>
                <CardContent className="space-y-3 text-xs">
                  <div className="flex justify-between border-b border-[#E6D6C6] pb-2">
                    <span className="text-[#806D60]">Adapter ID</span>
                    <span className="font-mono font-medium text-[#38271F]">
                      {stats.dataset_id}
                    </span>
                  </div>
                  <div className="flex justify-between border-b border-[#E6D6C6] pb-2">
                    <span className="text-[#806D60]">Modality</span>
                    <Badge variant="default">{stats.modality}</Badge>
                  </div>
                  <div className="flex justify-between border-b border-[#E6D6C6] pb-2">
                    <span className="text-[#806D60]">Supported Models</span>
                    <span className="font-mono text-[#38271F]">
                      {activeMeta?.supported_models.join(", ") || "—"}
                    </span>
                  </div>
                  <div className="flex justify-between border-b border-[#E6D6C6] pb-2">
                    <span className="text-[#806D60]">Raw Text Storage</span>
                    <span className="font-mono text-[#38271F]">
                      {activeMeta?.supports_raw_text ? "Yes" : "64-d Embeddings Only"}
                    </span>
                  </div>
                  <p className="leading-relaxed text-[#806D60]">{activeMeta?.description}</p>

                  <div className="rounded-xl border border-dashed border-[#8C6D58] bg-[#FAF3EC] p-4">
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-2 font-medium text-[#38271F]">
                        <UploadCloud className="h-5 w-5 text-[#6B3E26]" />
                        <span>Upload Custom Dataset</span>
                      </div>
                      <Badge variant="success">Active</Badge>
                    </div>
                    <p className="mt-2 text-xs text-[#806D60]">
                      Upload your own Image dataset (.zip with class folders or flat images) or Text dataset (.csv).
                    </p>
                    <Button
                      size="sm"
                      className="mt-3 w-full bg-[#6B3E26] hover:bg-[#54311E] text-white"
                      onClick={() => setUploadModalOpen(true)}
                    >
                      <UploadCloud className="mr-2 h-4 w-4" />
                      Upload New Dataset
                    </Button>
                  </div>
                </CardContent>
              </Card>
            </div>

            {/* Sample Preview Gallery */}
            <Card tone="secondary">
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
                      className="rounded-xl border border-[#E6D6C6] bg-[#FFFFFF] p-3.5 transition-all hover:border-[#6B3E26] hover:shadow-xs"
                    >
                      <div className="mb-2.5 flex items-center justify-between gap-2">
                        <span className="truncate font-mono text-[11px] font-medium text-[#38271F]">
                          {s.sample_id}
                        </span>
                        <Badge
                          variant={
                            s.split === "test"
                              ? "warning"
                              : s.status === "labeled"
                                ? "teal"
                                : "neutral"
                          }
                        >
                          {s.split}:{s.status}
                        </Badge>
                      </div>

                      {s.image_base64 ? (
                        <div className="mb-2.5 flex h-28 items-center justify-center rounded-lg border border-[#E6D6C6] bg-[#FAF3EC]">
                          <img
                            src={s.image_base64}
                            alt={s.sample_id}
                            className="h-24 w-24 rounded object-contain"
                            style={{ imageRendering: "pixelated" }}
                          />
                        </div>
                      ) : (
                        s.feature_summary && (
                          <div className="mb-2.5 rounded-lg border border-[#E6D6C6] bg-[#FAF3EC] p-2.5 font-mono text-[11px] text-[#38271F]">
                            <div>L2 Norm: {s.feature_summary.l2_norm}</div>
                            <div>
                              Mean/Std: {s.feature_summary.mean} / {s.feature_summary.std}
                            </div>
                            <div className="truncate text-[#806D60]">
                              Vec: [{s.embedding_preview?.slice(0, 5).join(", ")}...]
                            </div>
                          </div>
                        )
                      )}

                      <div className="flex items-center justify-between text-xs">
                        <span className="text-[#806D60]">True Class:</span>
                        <span className="font-semibold text-[#38271F]">
                          {s.true_class_name || "—"}
                        </span>
                      </div>
                      {s.annotator && (
                        <div className="mt-1 flex items-center justify-between text-[11px] text-[#806D60]">
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

      {/* Upload Custom Dataset Modal Dialog */}
      {uploadModalOpen && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 p-4">
          <div className="w-full max-w-lg rounded-2xl border border-[#E6D6C6] bg-[#FFFFFF] p-6 shadow-2xl">
            <div className="flex items-center justify-between border-b border-[#E6D6C6] pb-3">
              <div className="flex items-center gap-2">
                <UploadCloud className="h-5 w-5 text-[#6B3E26]" />
                <h3 className="text-lg font-semibold text-[#38271F]">
                  Upload Custom Dataset
                </h3>
              </div>
              <button
                className="text-[#806D60] hover:text-[#38271F] font-bold text-lg"
                onClick={() => setUploadModalOpen(false)}
              >
                ✕
              </button>
            </div>

            <form onSubmit={handleUploadSubmit} className="mt-4 space-y-4">
              <div>
                <label className="block text-xs font-semibold text-[#38271F] mb-1">
                  Dataset Name
                </label>
                <input
                  type="text"
                  placeholder="e.g. Medical X-Rays, Waste Sorting, Sentiment"
                  value={uploadName}
                  onChange={(e) => setUploadName(e.target.value)}
                  className="w-full rounded-lg border border-[#E6D6C6] px-3 py-2 text-sm text-[#38271F] focus:border-[#6B3E26] focus:outline-none"
                  required
                />
              </div>

              <div>
                <label className="block text-xs font-semibold text-[#38271F] mb-1">
                  Modality
                </label>
                <div className="flex gap-4">
                  <label className="flex items-center gap-2 text-sm text-[#38271F] cursor-pointer">
                    <input
                      type="radio"
                      name="modality"
                      value="image"
                      checked={uploadModality === "image"}
                      onChange={() => setUploadModality("image")}
                    />
                    Image Dataset (ZIP file)
                  </label>
                  <label className="flex items-center gap-2 text-sm text-[#38271F] cursor-pointer">
                    <input
                      type="radio"
                      name="modality"
                      value="text"
                      checked={uploadModality === "text"}
                      onChange={() => setUploadModality("text")}
                    />
                    Text Dataset (CSV file)
                  </label>
                </div>
              </div>

              <div>
                <label className="block text-xs font-semibold text-[#38271F] mb-1">
                  {uploadModality === "image" ? "Select .ZIP Archive" : "Select .CSV File"}
                </label>
                <input
                  type="file"
                  accept={uploadModality === "image" ? ".zip" : ".csv"}
                  onChange={(e) => {
                    if (e.target.files && e.target.files[0]) {
                      setUploadFile(e.target.files[0]);
                      if (!uploadName) {
                        const base = e.target.files[0].name.replace(/\.[^/.]+$/, "");
                        setUploadName(base.replace(/[_-]/g, " ").toUpperCase());
                      }
                    }
                  }}
                  className="w-full text-xs text-[#806D60] file:mr-3 file:py-2 file:px-4 file:rounded-lg file:border-0 file:text-xs file:font-semibold file:bg-[#FAF3EC] file:text-[#6B3E26] hover:file:bg-[#F3E7DC] cursor-pointer"
                  required
                />
                <p className="mt-1 text-[11px] text-[#806D60]">
                  {uploadModality === "image"
                    ? "Zip images grouped in class folders (e.g. dogs/*.jpg, cats/*.jpg) or flat images."
                    : "CSV with a 'text' column and an optional 'label' column."}
                </p>
              </div>

              {uploadMsg && (
                <div
                  className={`rounded-lg p-3 text-xs ${
                    uploadMsg.type === "success"
                      ? "bg-emerald-50 text-emerald-800 border border-emerald-200"
                      : "bg-red-50 text-red-800 border border-red-200"
                  }`}
                >
                  {uploadMsg.text}
                </div>
              )}

              <div className="flex justify-end gap-2 pt-2">
                <Button
                  type="button"
                  variant="outline"
                  onClick={() => setUploadModalOpen(false)}
                  disabled={uploading}
                >
                  Cancel
                </Button>
                <Button
                  type="submit"
                  className="bg-[#6B3E26] hover:bg-[#54311E] text-white"
                  disabled={uploading}
                >
                  {uploading ? "Processing & Normalizing..." : "Import Dataset"}
                </Button>
              </div>
            </form>
          </div>
        </div>
      )}
    </div>
  );
}

