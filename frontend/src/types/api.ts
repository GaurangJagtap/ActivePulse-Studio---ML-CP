export type NavSectionId =
  | "overview"
  | "datasets"
  | "lab"
  | "annotation"
  | "benchmarks"
  | "history";

export interface HealthResponse {
  status: string;
  service: string;
  version: string;
  pytorch_version: string;
  device: string;
  registered_strategies: number;
  label_studio_configured: boolean;
}

export interface DatasetSummary {
  id: string;
  name: string;
  modality: "image" | "text_embedding" | "text";
  num_classes: number;
  class_names: string[];
  default_model: string;
  supported_models: string[];
  supports_real_download: boolean;
  supports_raw_text: boolean;
  description: string;
}

export interface ClassCount {
  class_id: number;
  class_name: string;
  count: number;
}

export interface SplitStats {
  n_total: number;
  n_pool_unlabeled: number;
  n_queued: number;
  n_labeled: number;
  n_val: number;
  n_test: number;
}

export interface DatasetStatsResponse {
  dataset_id: string;
  name: string;
  modality: "image" | "text_embedding";
  use_synthetic: boolean;
  max_samples: number;
  num_classes: number;
  class_names: string[];
  split_stats: SplitStats;
  class_distribution: ClassCount[];
  feature_shape: number[];
  upload_supported: boolean;
  upload_status_note: string;
}

export interface DatasetSamplePreview {
  sample_id: string;
  split: string;
  status: string;
  true_class_id: number | null;
  true_class_name: string | null;
  assigned_class_name: string | null;
  annotator: string | null;
  image_base64: string | null;
  feature_summary: Record<string, number> | null;
  embedding_preview: number[] | null;
}

export interface DatasetSamplesResponse {
  dataset_id: string;
  total: number;
  offset: number;
  limit: number;
  samples: DatasetSamplePreview[];
}

export interface ModelInfo {
  id: string;
  name: string;
  architecture: string;
  supported_datasets: string[];
  embedding_dim: number;
  supports_mc_dropout: boolean;
  description: string;
}

export interface StrategyInfo {
  id: string;
  name: string;
  category: string;
  formula: string;
  description: string;
  requires_embeddings: boolean;
  requires_mc_dropout: boolean;
}

export type ExperimentMode = "simulated_oracle" | "human_in_the_loop";
export type ExperimentStatus =
  | "created"
  | "running"
  | "awaiting_annotations"
  | "completed"
  | "failed";

export interface ExperimentCreateRequest {
  name?: string;
  dataset: string;
  model_type: string;
  strategy: string;
  mode: ExperimentMode;
  initial_size: number;
  batch_size: number;
  max_rounds: number;
  label_budget?: number | null;
  target_accuracy?: number | null;
  max_samples: number;
  use_synthetic: boolean;
  max_epochs: number;
  lr: number;
  seed: number;
  label_noise_rate: number;
  warm_start: boolean;
  auto_start: boolean;
}

export interface RoundMetric {
  round_num: number;
  strategy: string;
  n_labeled: number;
  n_pool: number;
  test_acc: number;
  test_f1: number;
  ece: number | null;
  temperature: number | null;
  timestamp: string | null;
}

export interface CostSavings {
  total_samples: number;
  n_labeled: number;
  pct_dataset_labeled: number;
  estimated_passive_labels: number;
  estimated_labels_saved: number;
  estimated_hours_saved: number;
  estimated_dollars_saved: number;
  methodology_note: string;
}

export interface ExperimentSummary {
  id: string;
  name: string;
  dataset: string;
  model_type: string;
  strategy: string;
  mode: ExperimentMode;
  status: ExperimentStatus;
  current_round: number;
  max_rounds: number;
  current_stage: string;
  latest_accuracy: number | null;
  latest_f1: number | null;
  latest_ece: number | null;
  n_labeled: number;
  n_pool: number;
  n_queued: number;
  created_at: string;
  updated_at: string;
  error_message: string | null;
}

export interface ExperimentDetailResponse {
  summary: ExperimentSummary;
  config: Record<string, unknown>;
  rounds: RoundMetric[];
  cost_savings: CostSavings | null;
  stop_reason: string | null;
  logs: string[];
}

export interface ExperimentProgressResponse {
  experiment_id: string;
  status: ExperimentStatus;
  current_round: number;
  max_rounds: number;
  current_stage: string;
  stage_index: number;
  stages: string[];
  n_labeled: number;
  n_pool: number;
  n_queued: number;
  latest_accuracy: number | null;
  latest_f1: number | null;
  latest_ece: number | null;
  error_message: string | null;
  logs: string[];
}

export interface PcaPoint {
  sample_id: string;
  pc1: number;
  pc2: number;
  category:
    | "Unlabeled Pool"
    | "Already Labeled"
    | "Newly Queried by Active Learning";
}

export interface PcaEmbeddingResponse {
  experiment_id: string;
  round_num: number;
  explained_variance_ratio: number[];
  points: PcaPoint[];
}

export interface SelectedSample {
  sample_id: string;
  predicted_class_id: number;
  predicted_class_name: string;
  confidence: number;
  uncertainty: number;
  entropy: number;
  probabilities: number[];
  class_names: string[];
  image_base64: string | null;
  feature_summary: Record<string, number> | null;
  true_class_id: number | null;
  true_class_name: string | null;
}

export interface SelectedSamplesResponse {
  experiment_id: string;
  round_num: number;
  strategy: string;
  samples: SelectedSample[];
}

export interface PendingAnnotationSample {
  sample_id: string;
  experiment_id: string;
  dataset: string;
  modality: "image" | "text_embedding";
  round_num: number;
  predicted_class_id: number | null;
  predicted_class_name: string | null;
  confidence: number | null;
  uncertainty: number | null;
  probabilities: number[] | null;
  class_names: string[];
  image_base64: string | null;
  feature_summary: Record<string, number> | null;
  embedding_preview: number[] | null;
}

export interface PendingAnnotationsResponse {
  experiment_id: string;
  experiment_name: string;
  dataset: string;
  round_num: number;
  pending_count: number;
  completed_human_count: number;
  total_labeled_count: number;
  samples: PendingAnnotationSample[];
}

export interface AnnotationSubmitResponse {
  experiment_id: string;
  sample_id: string;
  assigned_label_id: number;
  assigned_label_name: string;
  annotator: string;
  remaining_pending: number;
  ready_to_retrain: boolean;
  is_correction?: boolean;
  continuous_learning_triggered?: boolean;
  message?: string | null;
}

export interface ContinuousModelMetrics {
  accuracy: number | null;
  macro_f1: number | null;
  ece: number | null;
  nll?: number | null;
}

export interface ContinuousRetrainEventItem {
  event_id: string;
  trigger_type: string;
  trigger_sample_id: string | null;
  samples_incorporated: number;
  total_training_samples: number;
  prev_model_version: string | null;
  candidate_model_version: string | null;
  prev_metrics: ContinuousModelMetrics;
  candidate_metrics: ContinuousModelMetrics;
  status: "pending" | "training" | "evaluating" | "promoted" | "rejected" | "failed";
  decision_reason: string | null;
  created_at: string | null;
  completed_at: string | null;
}

export interface ContinuousLearningStatusResponse {
  experiment_id: string;
  experiment_name: string;
  dataset: string;
  model_type: string;
  total_human_verified: number;
  newly_corrected: number;
  confirmed_predictions: number;
  samples_awaiting_training: number;
  total_training_samples: number;
  retraining_status:
    | "waiting_for_feedback"
    | "feedback_received"
    | "preparing_data"
    | "retraining_in_progress"
    | "evaluating_candidate"
    | "model_updated_successfully"
    | "candidate_rejected"
    | "retraining_failed";
  status_message: string;
  active_model_version: string;
  previous_model_version: string | null;
  latest_candidate_version: string | null;
  last_successful_update: string | null;
  active_metrics: ContinuousModelMetrics;
  previous_metrics: ContinuousModelMetrics | null;
  delta_metrics: Record<string, number> | null;
  history: ContinuousRetrainEventItem[];
}

export interface AnnotationHistoryItem {
  sample_id: string;
  experiment_id: string;
  assigned_label_id: number;
  assigned_label_name: string;
  true_label_id: number | null;
  true_label_name: string | null;
  matches_ground_truth: boolean | null;
  round_added: number | null;
  annotator: string;
  source_type: "human" | "simulated_oracle" | "cold_start" | "other";
  updated_at: string | null;
}

export interface AnnotationHistoryResponse {
  experiment_id: string;
  total_count: number;
  human_count: number;
  oracle_count: number;
  items: AnnotationHistoryItem[];
}

export interface LabelStudioStatusResponse {
  integration_status: "pending_phase_3";
  label_studio_configured: boolean;
  webhook_endpoint: string;
  supported_events: string[];
  notes: string;
}

export interface BenchmarkRunRequest {
  dataset: "cifar10" | "agnews";
  strategies: string[];
  seeds: number[];
  max_samples: number;
  initial_size: number;
  batch_size: number;
  rounds: number;
  epochs: number;
  target_accuracy: number;
  use_synthetic: boolean;
}

export interface BenchmarkSavingsRow {
  strategy: string;
  seeds: number;
  alc_mean: number;
  alc_ci95: number;
  alc_display: string;
  labels_to_target: number | null;
  labels_to_target_display: string;
  label_reduction_pct: number | null;
  label_reduction_display: string;
  final_accuracy: number | null;
  final_f1: number | null;
  final_ece: number | null;
}

export interface BenchmarkCurvePoint {
  strategy: string;
  n_labeled: number;
  mean_acc: number;
  ci95_acc: number;
  mean_f1: number | null;
  mean_ece: number | null;
}

export interface BenchmarkResultsResponse {
  source: string;
  dataset: string | null;
  target_accuracy: number | null;
  seeds: number[];
  has_learning_curve_image: boolean;
  summary_rows: BenchmarkSavingsRow[];
  curve_points: BenchmarkCurvePoint[];
}

export interface BenchmarkJobStatus {
  job_id: string;
  status: "running" | "completed" | "failed";
  current_strategy: string | null;
  current_seed: number | null;
  completed_steps: number;
  total_steps: number;
  error_message: string | null;
  started_at: string;
  completed_at: string | null;
}

