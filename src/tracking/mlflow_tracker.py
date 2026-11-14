"""MLflow experiment tracking wrapper with graceful fallback for ActivePulse Studio."""
from __future__ import annotations

import json
import os
import tempfile
from contextlib import contextmanager
from typing import Any, Dict, Iterator, List, Optional, Sequence


class MLflowTracker:
    """Tracks ActivePulse active-learning runs, per-round metrics, artifacts, and comparisons.

    Respects `MLFLOW_TRACKING_URI`, `MLFLOW_EXPERIMENT_NAME`, and `MLFLOW_ENABLED`.
    If tracking is disabled or an external tracking server is unreachable, operations
    degrade cleanly without interrupting training loops.
    """

    def __init__(
        self,
        tracking_uri: Optional[str] = None,
        experiment_name: Optional[str] = None,
        enabled: Optional[bool] = None,
    ):
        env_uri = os.environ.get("MLFLOW_TRACKING_URI", "").strip()
        env_flag = os.environ.get("MLFLOW_ENABLED", "").strip().lower()
        if enabled is not None:
            self.enabled = bool(enabled)
        elif env_flag in ("0", "false", "no"):
            self.enabled = False
        else:
            # Activate implicitly when an explicit tracking URI or MLFLOW_ENABLED=true is set
            self.enabled = bool(tracking_uri or env_uri or env_flag in ("1", "true", "yes"))

        self.tracking_uri = (
            tracking_uri
            or env_uri
            or f"sqlite:///{os.path.abspath(os.path.join('.', '.activepulse_data', 'mlflow.db')).replace(os.sep, '/')}"
        )
        self.experiment_name = (
            experiment_name
            or os.environ.get("MLFLOW_EXPERIMENT_NAME", "").strip()
            or "ActivePulse-Studio"
        )
        self._mlflow = None
        if self.enabled:
            try:
                if self.tracking_uri.startswith("sqlite:///"):
                    db_path = self.tracking_uri[len("sqlite:///") :]
                    parent = os.path.dirname(os.path.abspath(db_path))
                    if parent:
                        os.makedirs(parent, exist_ok=True)
                elif self.tracking_uri.startswith("file:///"):
                    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

                import mlflow

                os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
                mlflow.set_tracking_uri(self.tracking_uri)
                mlflow.set_experiment(self.experiment_name)
                self._mlflow = mlflow
            except Exception:
                self._mlflow = None

    @property
    def is_active(self) -> bool:
        return self.enabled and self._mlflow is not None

    @contextmanager
    def start_run(
        self,
        run_name: str,
        tags: Optional[Dict[str, str]] = None,
        nested: bool = False,
    ) -> Iterator[Optional[str]]:
        """Context manager starting an MLflow run and yielding its run_id (or None if disabled)."""
        if not self.is_active:
            yield None
            return
        try:
            with self._mlflow.start_run(run_name=run_name, tags=tags, nested=nested) as run:
                yield run.info.run_id
        except Exception:
            yield None

    def log_params(self, params: Dict[str, Any]) -> None:
        """Log hyperparameters and dataset/strategy configuration safely."""
        if not self.is_active or not params:
            return
        try:
            sanitized = {}
            for k, v in params.items():
                if v is None:
                    continue
                if isinstance(v, (dict, list)):
                    sanitized[str(k)] = json.dumps(v)[:250]
                else:
                    sanitized[str(k)] = str(v)[:250]
            if sanitized:
                self._mlflow.log_params(sanitized)
        except Exception:
            pass

    def log_round_metrics(
        self,
        round_num: int,
        accuracy: float,
        macro_f1: float,
        ece: float,
        n_labeled: int,
        n_pool: int,
        nll: Optional[float] = None,
        alc: Optional[float] = None,
        train_time_sec: Optional[float] = None,
    ) -> None:
        """Log per-round active-learning metrics indexed by active-learning round step."""
        if not self.is_active:
            return
        payload: Dict[str, float] = {
            "accuracy": float(accuracy),
            "macro_f1": float(macro_f1),
            "ece": float(ece),
            "n_labeled": float(n_labeled),
            "n_pool": float(n_pool),
        }
        if nll is not None:
            payload["nll"] = float(nll)
        if alc is not None:
            payload["alc"] = float(alc)
        if train_time_sec is not None:
            payload["train_time_sec"] = float(train_time_sec)
        try:
            self._mlflow.log_metrics(payload, step=int(round_num))
        except Exception:
            pass

    def log_summary_metrics(self, metrics: Dict[str, Any]) -> None:
        """Log aggregate run metrics (ALC, labels_to_target, label_reduction_pct, etc.)."""
        if not self.is_active or not metrics:
            return
        numeric = {
            str(k): float(v)
            for k, v in metrics.items()
            if isinstance(v, (int, float)) and not isinstance(v, bool)
        }
        if not numeric:
            return
        try:
            self._mlflow.log_metrics(numeric)
        except Exception:
            pass

    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None) -> None:
        """Log a local file (plot, CSV, checkpoint, config) as an MLflow artifact."""
        if not self.is_active or not local_path or not os.path.exists(local_path):
            return
        try:
            self._mlflow.log_artifact(local_path, artifact_path=artifact_path)
        except Exception:
            pass

    def log_dict_artifact(
        self,
        data: Dict[str, Any],
        filename: str,
        artifact_path: Optional[str] = None,
    ) -> None:
        """Serialize a dict as JSON and upload as an MLflow artifact (never pass hidden GT)."""
        if not self.is_active:
            return
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                target = os.path.join(tmpdir, filename)
                with open(target, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2)
                self._mlflow.log_artifact(target, artifact_path=artifact_path)
        except Exception:
            pass

    def compare_runs(
        self,
        experiment_ids: Optional[Sequence[str]] = None,
        max_results: int = 50,
    ) -> List[Dict[str, Any]]:
        """Query tracked runs for cross-experiment / cross-seed comparison."""
        if not self.is_active:
            return []
        try:
            exp = self._mlflow.get_experiment_by_name(self.experiment_name)
            if exp is None:
                return []
            df = self._mlflow.search_runs(
                experiment_ids=[exp.experiment_id],
                max_results=max_results,
            )
            records: List[Dict[str, Any]] = []
            for _, row in df.iterrows():
                rec: Dict[str, Any] = {
                    "run_id": row.get("run_id"),
                    "run_name": row.get("tags.mlflow.runName"),
                    "status": row.get("status"),
                }
                for col in df.columns:
                    val = row[col]
                    if val is None or (isinstance(val, float) and val != val):
                        continue
                    if col.startswith("params.") or col.startswith("metrics."):
                        rec[col] = val
                    elif col == "tags.activepulse_experiment_id":
                        rec["experiment_id"] = val
                if experiment_ids and rec.get("experiment_id") not in experiment_ids:
                    continue
                records.append(rec)
            return records
        except Exception:
            return []

