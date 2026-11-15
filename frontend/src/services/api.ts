import type {
  AnnotationHistoryResponse,
  AnnotationSubmitResponse,
  BenchmarkJobStatus,
  BenchmarkResultsResponse,
  BenchmarkRunRequest,
  ContinuousLearningStatusResponse,
  DatasetSamplesResponse,
  DatasetStatsResponse,
  DatasetSummary,
  ExperimentCreateRequest,
  ExperimentDetailResponse,
  ExperimentProgressResponse,
  ExperimentSummary,
  HealthResponse,
  LabelStudioStatusResponse,
  ModelInfo,
  PcaEmbeddingResponse,
  PendingAnnotationsResponse,
  SelectedSamplesResponse,
  StrategyInfo,
} from "@/types/api";

const API_BASE =
  import.meta.env.VITE_API_BASE_URL?.replace(/\/$/, "") || "/api/v1";

export class ApiError extends Error {
  status: number;
  constructor(message: string, status: number) {
    super(message);
    this.status = status;
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    headers: {
      "Content-Type": "application/json",
      ...(init?.headers || {}),
    },
    ...init,
  });

  if (!res.ok) {
    let detail = `Request failed (${res.status})`;
    try {
      const data = await res.json();
      detail = data.detail || data.message || detail;
    } catch {
      // ignore json parse failure
    }
    throw new ApiError(detail, res.status);
  }

  return res.json() as Promise<T>;
}

export const api = {
  health: () => request<HealthResponse>("/health"),

  // Catalog
  listDatasets: () => request<DatasetSummary[]>("/datasets"),
  listModels: () => request<ModelInfo[]>("/models"),
  listStrategies: () => request<StrategyInfo[]>("/strategies"),

  // Datasets
  getDatasetStats: (
    datasetId: string,
    params: { max_samples?: number; use_synthetic?: boolean; seed?: number; experiment_id?: string } = {}
  ) => {
    const qs = new URLSearchParams();
    if (params.max_samples) qs.set("max_samples", String(params.max_samples));
    if (params.use_synthetic !== undefined) qs.set("use_synthetic", String(params.use_synthetic));
    if (params.seed !== undefined) qs.set("seed", String(params.seed));
    if (params.experiment_id) qs.set("experiment_id", params.experiment_id);
    return request<DatasetStatsResponse>(`/datasets/${datasetId}/stats?${qs.toString()}`);
  },

  getDatasetSamples: (
    datasetId: string,
    params: {
      offset?: number;
      limit?: number;
      max_samples?: number;
      use_synthetic?: boolean;
      split?: string;
      status?: string;
      experiment_id?: string;
    } = {}
  ) => {
    const qs = new URLSearchParams();
    if (params.offset !== undefined) qs.set("offset", String(params.offset));
    if (params.limit !== undefined) qs.set("limit", String(params.limit));
    if (params.max_samples) qs.set("max_samples", String(params.max_samples));
    if (params.use_synthetic !== undefined) qs.set("use_synthetic", String(params.use_synthetic));
    if (params.split) qs.set("split", params.split);
    if (params.status) qs.set("status", params.status);
    if (params.experiment_id) qs.set("experiment_id", params.experiment_id);
    return request<DatasetSamplesResponse>(`/datasets/${datasetId}/samples?${qs.toString()}`);
  },

  // Experiments
  listExperiments: () => request<ExperimentSummary[]>("/experiments"),
  createExperiment: (payload: ExperimentCreateRequest) =>
    request<ExperimentSummary>("/experiments", {
      method: "POST",
      body: JSON.stringify(payload),
    }),
  getExperiment: (id: string) => request<ExperimentDetailResponse>(`/experiments/${id}`),
  startExperiment: (id: string) =>
    request<ExperimentSummary>(`/experiments/${id}/start`, { method: "POST" }),
  advanceExperiment: (id: string) =>
    request<ExperimentSummary>(`/experiments/${id}/advance`, { method: "POST" }),
  getExperimentProgress: (id: string) =>
    request<ExperimentProgressResponse>(`/experiments/${id}/progress`),
  getExperimentPca: (id: string) =>
    request<PcaEmbeddingResponse>(`/experiments/${id}/pca`),
  getExperimentSelectedSamples: (id: string) =>
    request<SelectedSamplesResponse>(`/experiments/${id}/selected-samples`),
  deleteExperiment: (id: string) =>
    request<{ deleted: boolean }>(`/experiments/${id}`, { method: "DELETE" }),
  getExperimentExportUrl: (id: string) => `${API_BASE}/experiments/${id}/export`,

  // Annotations
  getPendingAnnotations: (experimentId: string) =>
    request<PendingAnnotationsResponse>(
      `/annotations/pending?experiment_id=${encodeURIComponent(experimentId)}`
    ),
  submitAnnotation: (payload: {
    experiment_id: string;
    sample_id: string;
    label_id: number;
    annotator: string;
  }) =>
    request<AnnotationSubmitResponse>("/annotations/submit", {
      method: "POST",
      body: JSON.stringify(payload),
    }),
  getAnnotationHistory: (experimentId: string) =>
    request<AnnotationHistoryResponse>(
      `/annotations/history?experiment_id=${encodeURIComponent(experimentId)}`
    ),
  getLabelStudioStatus: () =>
    request<LabelStudioStatusResponse>("/webhooks/label-studio/status"),
  getContinuousLearningStatus: (experimentId: string) =>
    request<ContinuousLearningStatusResponse>(
      `/continuous-learning/status?experiment_id=${encodeURIComponent(experimentId)}`
    ),
  triggerContinuousRetrain: (experimentId: string, force = false) =>
    request<{ status: string; job_id: string; message: string }>(
      "/continuous-learning/retrain",
      {
        method: "POST",
        body: JSON.stringify({ experiment_id: experimentId, force }),
      }
    ),

  // Benchmarks
  getBenchmarkResults: () => request<BenchmarkResultsResponse>("/benchmarks/results"),
  runBenchmark: (payload: BenchmarkRunRequest) =>
    request<BenchmarkJobStatus>("/benchmarks/run", {
      method: "POST",
      body: JSON.stringify(payload),
    }),
  getBenchmarkStatus: (jobId: string) =>
    request<BenchmarkJobStatus>(`/benchmarks/status/${jobId}`),
  getBenchmarkExportUrl: () => `${API_BASE}/benchmarks/export`,
  getBenchmarkPlotUrl: () => `${API_BASE}/benchmarks/plot`,
};

