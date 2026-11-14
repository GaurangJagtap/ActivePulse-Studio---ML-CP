"""Label Studio SDK integration for human-in-the-loop annotation and model pre-annotations."""
from __future__ import annotations

import base64
import io
import os
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple
from xml.sax.saxutils import escape

import numpy as np
import torch

from src.data.db import DatabaseManager
from src.labeling.base import BaseLabeler
from src.labeling.quality import select_double_annotation_samples
from src.models.wrapper import ActiveLearningClassifier


class PendingHumanAnnotationError(RuntimeError):
    """Raised when a synchronous loop requests labels that are queued for human review."""


@dataclass
class LabelStudioConnectionConfig:
    """Environment-backed connection settings for Label Studio."""

    base_url: str
    api_key: str
    webhook_secret: str = ""
    default_project_id: Optional[int] = None

    @classmethod
    def from_env(cls) -> "LabelStudioConnectionConfig":
        raw_pid = os.environ.get("LABEL_STUDIO_PROJECT_ID", "").strip()
        project_id = int(raw_pid) if raw_pid.isdigit() else None
        return cls(
            base_url=os.environ.get("LABEL_STUDIO_URL", "").strip().rstrip("/"),
            api_key=os.environ.get("LABEL_STUDIO_API_KEY", "").strip(),
            webhook_secret=os.environ.get("LABEL_STUDIO_WEBHOOK_SECRET", "").strip(),
            default_project_id=project_id,
        )

    @property
    def is_configured(self) -> bool:
        return bool(self.base_url and self.api_key)


def build_image_classification_config(
    class_names: Sequence[str],
    header: str = "Select the best matching image category",
) -> str:
    """Generate a Label Studio XML labeling configuration for image classification."""
    choices = "\n".join(f'    <Choice value="{escape(str(name))}"/>' for name in class_names)
    return (
        "<View>\n"
        f'  <Header value="{escape(header)}"/>\n'
        '  <Image name="image" value="$image" zoom="true"/>\n'
        '  <Choices name="label" toName="image" choice="single" showInLine="true" required="true">\n'
        f"{choices}\n"
        "  </Choices>\n"
        "</View>"
    )


def build_text_classification_config(
    class_names: Sequence[str],
    header: str = "Select the best matching topic category",
) -> str:
    """Generate a Label Studio XML labeling configuration for text classification."""
    choices = "\n".join(f'    <Choice value="{escape(str(name))}"/>' for name in class_names)
    return (
        "<View>\n"
        f'  <Header value="{escape(header)}"/>\n'
        '  <Text name="text" value="$text"/>\n'
        '  <Choices name="label" toName="text" choice="single" showInLine="true" required="true">\n'
        f"{choices}\n"
        "  </Choices>\n"
        "</View>"
    )


def build_labeling_config(modality: str, class_names: Sequence[str]) -> str:
    """Dispatch labeling XML template by dataset modality ('image' or 'text')."""
    if modality.lower() == "image":
        return build_image_classification_config(class_names)
    return build_text_classification_config(class_names)


def tensor_to_png_data_uri(feature: Any) -> str:
    """Encode an image tensor/ndarray (CHW or HWC) as a base64 PNG data URI for Label Studio."""
    from PIL import Image

    arr = feature.detach().cpu().numpy() if isinstance(feature, torch.Tensor) else np.asarray(feature)
    if arr.ndim == 3 and arr.shape[0] in (1, 3, 4):
        arr = np.transpose(arr, (1, 2, 0))
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)

    # Normalize normalized float tensors cleanly into uint8 [0, 255]
    if np.issubdtype(arr.dtype, np.floating):
        min_v = float(arr.min())
        max_v = float(arr.max())
        if min_v < 0.0 or max_v > 1.0:
            # Handle ImageNet/CIFAR normalized tensors
            denom = max(max_v - min_v, 1e-6)
            arr = (arr - min_v) / denom
        arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    else:
        arr = np.clip(arr, 0, 255).astype(np.uint8)

    img = Image.fromarray(arr).resize((160, 160), resample=Image.NEAREST)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def generate_model_pre_annotations(
    model_wrapper: ActiveLearningClassifier,
    sample_ids: Sequence[str],
    features: Any,
    class_names: Sequence[str],
) -> List[Dict[str, Any]]:
    """Generate model predictions and calibrated confidences for queried samples.

    Pre-annotations are advisory hints for human reviewers and are NEVER stored
    as confirmed labels.
    """
    if len(sample_ids) == 0:
        return []

    feat_arr = features.detach().cpu().numpy() if isinstance(features, torch.Tensor) else np.asarray(features)
    if not getattr(model_wrapper, "initialized_", False):
        return []
    probs = model_wrapper.predict_calibrated_proba(feat_arr)
    pre_annotations: List[Dict[str, Any]] = []

    for idx, sid in enumerate(sample_ids):
        row = probs[idx]
        pred_id = int(np.argmax(row))
        confidence = float(row[pred_id])
        pred_name = class_names[pred_id] if 0 <= pred_id < len(class_names) else str(pred_id)
        pre_annotations.append(
            {
                "sample_id": str(sid),
                "predicted_label": pred_id,
                "predicted_label_name": pred_name,
                "confidence": round(confidence, 6),
                "probabilities": [round(float(p), 6) for p in row.tolist()],
            }
        )
    return pre_annotations


def extract_label_from_ls_result(
    result: Any,
    class_names: Sequence[str],
) -> Optional[int]:
    """Extract a validated integer class index from a Label Studio annotation result list."""
    if not isinstance(result, list):
        return None

    name_to_idx = {str(name).strip().lower(): idx for idx, name in enumerate(class_names)}
    for item in result:
        if not isinstance(item, dict):
            continue
        value = item.get("value") or {}
        choices = value.get("choices")
        if isinstance(choices, list) and choices:
            chosen = str(choices[0]).strip().lower()
            if chosen in name_to_idx:
                return name_to_idx[chosen]
            if chosen.isdigit():
                idx = int(chosen)
                if 0 <= idx < len(class_names):
                    return idx
    return None


class LabelStudioManager:
    """High-level coordinator for Label Studio projects, tasks, pre-annotations, and sync."""

    def __init__(
        self,
        config: Optional[LabelStudioConnectionConfig] = None,
        client: Optional[Any] = None,
    ):
        self.config = config or LabelStudioConnectionConfig.from_env()
        self._client = client

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        if not self.config.is_configured:
            raise RuntimeError(
                "Label Studio is not configured. Set LABEL_STUDIO_URL and LABEL_STUDIO_API_KEY."
            )
        from label_studio_sdk.client import LabelStudio

        self._client = LabelStudio(
            base_url=self.config.base_url,
            api_key=self.config.api_key,
        )
        return self._client

    def health_status(self) -> Dict[str, Any]:
        """Inspect configuration and live connectivity to Label Studio without raising."""
        if not self.config.is_configured and self._client is None:
            return {
                "integration_status": "unconfigured",
                "configured": False,
                "connected": False,
                "base_url": self.config.base_url or None,
                "default_project_id": self.config.default_project_id,
                "error": "LABEL_STUDIO_URL and LABEL_STUDIO_API_KEY are not configured.",
            }
        try:
            client = self._get_client()
            projects = list(client.projects.list())
            return {
                "integration_status": "connected",
                "configured": True,
                "connected": True,
                "base_url": self.config.base_url or "mock://label-studio",
                "default_project_id": self.config.default_project_id,
                "project_count": len(projects),
                "error": None,
            }
        except Exception as exc:
            return {
                "integration_status": "configured",
                "configured": True,
                "connected": False,
                "base_url": self.config.base_url or None,
                "default_project_id": self.config.default_project_id,
                "project_count": 0,
                "error": str(exc),
            }

    def ensure_project(
        self,
        title: str,
        modality: str,
        class_names: Sequence[str],
        description: Optional[str] = None,
        overlap_rate: float = 0.10,
        webhook_url: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Find an existing Label Studio project by title or create it automatically."""
        client = self._get_client()
        existing = None
        for proj in client.projects.list():
            proj_title = getattr(proj, "title", None) if not isinstance(proj, dict) else proj.get("title")
            if proj_title == title:
                existing = proj
                break

        if existing is not None:
            pid = int(getattr(existing, "id", None) if not isinstance(existing, dict) else existing["id"])
            return {"project_id": pid, "title": title, "created": False}

        label_config = build_labeling_config(modality=modality, class_names=class_names)
        overlap_pct = max(0, min(100, int(round(overlap_rate * 100))))
        created = client.projects.create(
            title=title,
            description=description or f"ActivePulse Studio ({modality}) active-learning project",
            label_config=label_config,
            show_collab_predictions=True,
            show_annotation_history=True,
            maximum_annotations=2,
            overlap_cohort_percentage=overlap_pct,
        )
        pid = int(getattr(created, "id", None) if not isinstance(created, dict) else created["id"])

        if webhook_url:
            headers = {}
            if self.config.webhook_secret:
                headers["X-Label-Studio-Secret"] = self.config.webhook_secret
            client.webhooks.create(
                url=webhook_url,
                project=pid,
                actions=["ANNOTATION_CREATED", "ANNOTATION_UPDATED"],
                headers=headers or None,
                send_payload=True,
                is_active=True,
            )

        return {"project_id": pid, "title": title, "created": True, "label_config": label_config}

    def create_tasks_with_pre_annotations(
        self,
        project_id: int,
        samples: Sequence[Dict[str, Any]],
        class_names: Sequence[str],
        db: DatabaseManager,
        pre_annotations: Optional[Sequence[Dict[str, Any]]] = None,
        batch_id: Optional[str] = None,
        double_annotation_ids: Optional[Sequence[str]] = None,
        model_version: str = "activepulse-v1",
        experiment_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Upload queried samples to Label Studio, attach model predictions, and persist mappings.

        Never embeds ground-truth labels inside uploaded task payloads or metadata.
        """
        client = self._get_client()
        pre_map = {str(item["sample_id"]): item for item in (pre_annotations or [])}
        double_set = set(str(s) for s in (double_annotation_ids or []))

        created_mappings: List[Dict[str, Any]] = []
        db_pre_records: List[Dict[str, Any]] = []

        for sample in samples:
            sid = str(sample["sample_id"])
            modality = str(sample.get("modality", "image")).lower()

            # Build safe Label Studio task payload without ground-truth fields
            if modality == "image":
                img_value = sample.get("image_uri") or tensor_to_png_data_uri(sample["feature"])
                task_data = {"image": img_value, "sample_id": sid}
            else:
                text_value = sample.get("text") or sample.get("preview_text") or f"Sample {sid}"
                task_data = {"text": str(text_value), "sample_id": sid}

            meta = {
                "sample_id": sid,
                "experiment_id": experiment_id or db.experiment_id,
                "batch_id": batch_id,
                "double_annotation": sid in double_set,
            }
            overlap = 2 if sid in double_set else 1

            task = client.tasks.create(
                project=project_id,
                data=task_data,
                meta=meta,
                overlap=overlap,
            )
            ls_task_id = int(getattr(task, "id", None) if not isinstance(task, dict) else task["id"])

            pre = pre_map.get(sid)
            if pre is not None:
                from_name = "label"
                to_name = "image" if modality == "image" else "text"
                pred_name = pre.get("predicted_label_name")
                if not pred_name:
                    pred_idx = int(pre["predicted_label"])
                    pred_name = class_names[pred_idx] if 0 <= pred_idx < len(class_names) else str(pred_idx)
                pred_result = [
                    {
                        "from_name": from_name,
                        "to_name": to_name,
                        "type": "choices",
                        "value": {"choices": [str(pred_name)]},
                    }
                ]
                client.predictions.create(
                    task=ls_task_id,
                    project=project_id,
                    result=pred_result,
                    score=float(pre["confidence"]),
                    model_version=model_version,
                )
                db_pre_records.append(
                    {
                        "sample_id": sid,
                        "predicted_label": int(pre["predicted_label"]),
                        "confidence": float(pre["confidence"]),
                        "probabilities": pre.get("probabilities"),
                        "task_id": ls_task_id,
                        "batch_id": batch_id,
                    }
                )
            else:
                db_pre_records.append(
                    {
                        "sample_id": sid,
                        "predicted_label": None,
                        "confidence": None,
                        "probabilities": None,
                        "task_id": ls_task_id,
                        "batch_id": batch_id,
                    }
                )

            created_mappings.append(
                {
                    "sample_id": sid,
                    "label_studio_task_id": ls_task_id,
                    "double_annotation": sid in double_set,
                    "has_pre_annotation": pre is not None,
                }
            )

        if db_pre_records:
            db.store_pre_annotations(db_pre_records)

        return {
            "project_id": project_id,
            "tasks_created": len(created_mappings),
            "double_annotation_count": len(double_set),
            "mappings": created_mappings,
        }

    def sync_completed_annotations(
        self,
        project_id: int,
        class_names: Sequence[str],
        db: DatabaseManager,
        round_num: int,
        batch_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Pull completed human annotations from Label Studio and store them idempotently."""
        client = self._get_client()
        tasks = list(client.tasks.list(project=project_id))

        synced_count = 0
        duplicate_count = 0
        skipped_invalid = 0
        synced_samples: List[str] = []

        for task in tasks:
            task_id = int(getattr(task, "id", None) if not isinstance(task, dict) else task["id"])
            task_meta = getattr(task, "meta", None) if not isinstance(task, dict) else task.get("meta")
            task_data = getattr(task, "data", None) if not isinstance(task, dict) else task.get("data")
            annotations = (
                getattr(task, "annotations", None)
                if not isinstance(task, dict)
                else task.get("annotations")
            )
            if annotations is None and hasattr(client, "annotations") and hasattr(client.annotations, "list"):
                try:
                    annotations = list(client.annotations.list(id=task_id))
                except Exception:
                    annotations = []

            sample_rec = db.get_sample_by_ls_task_id(task_id)
            sid: Optional[str] = sample_rec.sample_id if sample_rec else None
            if sid is None and isinstance(task_meta, dict) and task_meta.get("sample_id"):
                sid = str(task_meta["sample_id"])
            elif sid is None and isinstance(task_data, dict) and task_data.get("sample_id"):
                sid = str(task_data["sample_id"])

            if not sid or not annotations:
                continue

            for ann in annotations:
                was_cancelled = (
                    getattr(ann, "was_cancelled", False)
                    if not isinstance(ann, dict)
                    else bool(ann.get("was_cancelled", False))
                )
                if was_cancelled:
                    continue

                ann_id = getattr(ann, "id", None) if not isinstance(ann, dict) else ann.get("id")
                result = getattr(ann, "result", None) if not isinstance(ann, dict) else ann.get("result")
                label_id = extract_label_from_ls_result(result, class_names)
                if label_id is None:
                    skipped_invalid += 1
                    continue

                completed_by = (
                    getattr(ann, "completed_by", None)
                    if not isinstance(ann, dict)
                    else ann.get("completed_by")
                )
                if isinstance(completed_by, dict):
                    annotator_name = (
                        completed_by.get("email")
                        or completed_by.get("username")
                        or f"user_{completed_by.get('id', 'unknown')}"
                    )
                elif completed_by is not None:
                    annotator_name = f"ls_user_{completed_by}"
                else:
                    annotator_name = "label_studio_human"

                event_key = f"ls_sync_ann_{ann_id}" if ann_id is not None else None
                inserted, _ = db.record_human_annotation(
                    sample_id=sid,
                    label_id=label_id,
                    annotator=annotator_name,
                    round_num=round_num,
                    source_type="label_studio",
                    batch_id=batch_id,
                    webhook_event_id=event_key,
                    label_studio_annotation_id=int(ann_id) if ann_id is not None else None,
                    is_double_annotation=bool(
                        isinstance(task_meta, dict) and task_meta.get("double_annotation")
                    ),
                )
                if inserted:
                    synced_count += 1
                    synced_samples.append(sid)
                else:
                    duplicate_count += 1

        if batch_id:
            db.refresh_batch_progress(batch_id)

        return {
            "project_id": project_id,
            "synced_count": synced_count,
            "duplicate_skipped": duplicate_count,
            "invalid_skipped": skipped_invalid,
            "synced_sample_ids": synced_samples,
        }


class LabelStudioLabeler(BaseLabeler):
    """Human-in-the-loop labeler backed by Label Studio and internal persistence."""

    def __init__(
        self,
        db: DatabaseManager,
        dataset: Any,
        project_id: Optional[int] = None,
        manager: Optional[LabelStudioManager] = None,
        model_wrapper: Optional[ActiveLearningClassifier] = None,
        double_annotation_rate: float = 0.10,
        round_num_fn: Optional[Callable[[], int]] = None,
    ):
        self.db = db
        self.dataset = dataset
        self.manager = manager or LabelStudioManager()
        self.project_id = project_id or self.manager.config.default_project_id
        self.model_wrapper = model_wrapper
        self.double_annotation_rate = double_annotation_rate
        self.round_num_fn = round_num_fn

    def queue_and_upload_batch(
        self,
        sample_ids: Sequence[str],
        round_num: int,
        strategy: str,
        batch_id: Optional[str] = None,
        seed: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Generate pre-annotations and upload a queried batch to Label Studio."""
        if not sample_ids:
            return {"project_id": self.project_id, "tasks_created": 0, "mappings": []}

        bid = batch_id or f"{self.db.experiment_id}_r{round_num}"
        modality = "image" if str(getattr(self.dataset, "name", "")).lower() == "cifar10" else "text"
        if self.project_id is None:
            proj = self.manager.ensure_project(
                title=f"ActivePulse - {self.dataset.name} ({self.db.experiment_id})",
                modality=modality,
                class_names=self.dataset.class_names,
                overlap_rate=self.double_annotation_rate,
            )
            self.project_id = int(proj["project_id"])

        double_ids = select_double_annotation_samples(
            sample_ids,
            overlap_rate=self.double_annotation_rate,
            seed=seed,
        )

        # Gather features & previews without ground-truth labels
        features = self.dataset.get_features(list(sample_ids))
        pre_annotations = (
            generate_model_pre_annotations(
                self.model_wrapper, sample_ids, features, self.dataset.class_names
            )
            if self.model_wrapper is not None
            else []
        )

        sample_payloads: List[Dict[str, Any]] = []
        for idx, sid in enumerate(sample_ids):
            payload: Dict[str, Any] = {
                "sample_id": str(sid),
                "modality": modality,
                "feature": features[idx],
            }
            if modality == "text":
                if hasattr(self.dataset, "texts") and hasattr(self.dataset, "id_to_index"):
                    ds_idx = self.dataset.id_to_index.get(str(sid))
                    if ds_idx is not None and ds_idx < len(self.dataset.texts):
                        payload["text"] = self.dataset.texts[ds_idx]
                if "text" not in payload:
                    vec = np.asarray(features[idx]).reshape(-1)
                    preview = ", ".join(f"{float(v):.3f}" for v in vec[:8])
                    payload["text"] = f"Sample {sid} (embedding norm={float(np.linalg.norm(vec)):.3f}, dims[:8]=[{preview}])"
            sample_payloads.append(payload)

        threshold_count = len(sample_ids)
        self.db.create_or_update_batch(
            batch_id=bid,
            round_num=round_num,
            strategy=strategy,
            total_tasks=len(sample_ids),
            threshold_count=threshold_count,
            double_annotation_rate=self.double_annotation_rate,
        )

        return self.manager.create_tasks_with_pre_annotations(
            project_id=self.project_id,
            samples=sample_payloads,
            class_names=self.dataset.class_names,
            db=self.db,
            pre_annotations=pre_annotations,
            batch_id=bid,
            double_annotation_ids=double_ids,
            experiment_id=self.db.experiment_id,
        )

    def label(self, sample_ids: List[str]) -> List[int]:
        """Upload queried samples to Label Studio and return confirmed labels if already synced.

        If human annotations are still pending in Label Studio, raises
        `PendingHumanAnnotationError` so asynchronous workflows wait cleanly instead
        of exposing ground-truth labels.
        """
        round_num = self.round_num_fn() if self.round_num_fn else 0
        if self.project_id is not None:
            self.manager.sync_completed_annotations(
                project_id=self.project_id,
                class_names=self.dataset.class_names,
                db=self.db,
                round_num=round_num,
            )

        # Check which samples already have human annotations recorded in DB
        resolved: List[int] = []
        missing: List[str] = []
        for sid in sample_ids:
            rec = self.db.get_sample(sid)
            if rec is not None and rec.status == "labeled" and rec.label is not None:
                resolved.append(int(rec.label))
            else:
                missing.append(sid)

        if missing:
            self.queue_and_upload_batch(missing, round_num=round_num, strategy="human_in_loop")
            raise PendingHumanAnnotationError(
                f"Uploaded {len(missing)} sample(s) to Label Studio project {self.project_id}; "
                "awaiting human submission."
            )

        return resolved
