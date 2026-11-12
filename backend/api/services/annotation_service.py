"""Human-in-the-Loop Annotation service & Label Studio webhook readiness."""

from typing import List, Optional
import numpy as np

from backend.api.config import LABEL_STUDIO_URL
from backend.api.schemas.annotations import (
    AnnotationHistoryItem,
    AnnotationHistoryResponse,
    AnnotationSubmitRequest,
    AnnotationSubmitResponse,
    LabelStudioStatusResponse,
    LabelStudioWebhookPayload,
    PendingAnnotationSample,
    PendingAnnotationsResponse,
)
from backend.api.services.dataset_service import get_cached_dataset
from backend.api.services.experiment_service import (
    get_experiment_db,
    get_experiment_detail,
    get_experiment_runtime,
    _append_log,
    _update_meta,
    _get_db_counts,
)
from backend.api.services.utils import feature_to_base64_png, summarize_text_embedding
from src.data.db import SampleRecord


def get_pending_annotations(exp_id: str) -> PendingAnnotationsResponse:
    detail = get_experiment_detail(exp_id)
    summary = detail.summary
    cfg = detail.config

    dataset = get_cached_dataset(
        dataset_id=summary.dataset,
        max_samples=int(cfg.get("max_samples", 450)),
        use_synthetic=bool(cfg.get("use_synthetic", True)),
        seed=int(cfg.get("seed", 42)),
    )
    db = get_experiment_db(exp_id)
    runtime = get_experiment_runtime(exp_id)

    with db.get_session() as session:
        queued_rows = (
            session.query(SampleRecord.sample_id)
            .filter(SampleRecord.split == "pool", SampleRecord.status == "queued")
            .order_by(SampleRecord.sample_id.asc())
            .all()
        )
        queued_ids = [r[0] for r in queued_rows]

        human_count = (
            session.query(SampleRecord)
            .filter(
                SampleRecord.split == "pool",
                SampleRecord.status == "labeled",
                SampleRecord.annotator.like("human%"),
            )
            .count()
        )
        total_labeled = (
            session.query(SampleRecord)
            .filter(SampleRecord.split == "pool", SampleRecord.status == "labeled")
            .count()
        )

    samples: List[PendingAnnotationSample] = []
    if queued_ids:
        feats = dataset.get_features(queued_ids)
        probs = None
        if runtime is not None and getattr(runtime.orchestrator.model, "initialized_", False):
            probs = runtime.orchestrator.model.predict_calibrated_proba(feats)

        modality = "image" if dataset.name == "cifar10" else "text_embedding"
        for i, sid in enumerate(queued_ids):
            p_vec = probs[i] if probs is not None else None
            pred_id = int(np.argmax(p_vec)) if p_vec is not None else None
            conf = float(np.max(p_vec)) if p_vec is not None else None
            unc = (1.0 - conf) if conf is not None else None

            samples.append(
                PendingAnnotationSample(
                    sample_id=sid,
                    experiment_id=exp_id,
                    dataset=dataset.name,
                    modality=modality,
                    round_num=summary.current_round,
                    predicted_class_id=pred_id,
                    predicted_class_name=dataset.class_names[pred_id] if pred_id is not None else None,
                    confidence=round(conf, 4) if conf is not None else None,
                    uncertainty=round(unc, 4) if unc is not None else None,
                    probabilities=[round(float(v), 5) for v in p_vec] if p_vec is not None else None,
                    class_names=dataset.class_names,
                    image_base64=feature_to_base64_png(feats[i]) if dataset.name == "cifar10" else None,
                    feature_summary=summarize_text_embedding(feats[i]) if dataset.name == "agnews" else None,
                    embedding_preview=[round(float(v), 4) for v in feats[i].reshape(-1)[:16]]
                    if dataset.name == "agnews"
                    else None,
                )
            )

    return PendingAnnotationsResponse(
        experiment_id=exp_id,
        experiment_name=summary.name,
        dataset=dataset.name,
        round_num=summary.current_round,
        pending_count=len(samples),
        completed_human_count=human_count,
        total_labeled_count=total_labeled,
        samples=samples,
    )


def submit_annotation(req: AnnotationSubmitRequest) -> AnnotationSubmitResponse:
    detail = get_experiment_detail(req.experiment_id)
    summary = detail.summary
    cfg = detail.config

    dataset = get_cached_dataset(
        dataset_id=summary.dataset,
        max_samples=int(cfg.get("max_samples", 450)),
        use_synthetic=bool(cfg.get("use_synthetic", True)),
        seed=int(cfg.get("seed", 42)),
    )
    if req.label_id < 0 or req.label_id >= dataset.num_classes:
        raise ValueError(f"label_id must be in range [0, {dataset.num_classes - 1}].")

    db = get_experiment_db(req.experiment_id)
    with db.get_session() as session:
        record = session.get(SampleRecord, req.sample_id)
        if record is None or record.split != "pool":
            raise KeyError(f"Pool sample '{req.sample_id}' not found in experiment '{req.experiment_id}'.")

    annotator_tag = req.annotator if req.annotator.startswith("human") else f"human_{req.annotator}"
    db.mark_labeled(
        sample_ids=[req.sample_id],
        labels=[req.label_id],
        round_num=summary.current_round,
        annotator=annotator_tag,
    )

    counts = _get_db_counts(db)
    _update_meta(
        req.experiment_id,
        n_labeled=counts["n_labeled"],
        n_pool=counts["n_pool"],
        n_queued=counts["n_queued"],
    )
    _append_log(
        req.experiment_id,
        f"Annotated sample {req.sample_id} as '{dataset.class_names[req.label_id]}' ({annotator_tag}). "
        f"Pending queue: {counts['n_queued']}.",
    )

    return AnnotationSubmitResponse(
        experiment_id=req.experiment_id,
        sample_id=req.sample_id,
        assigned_label_id=req.label_id,
        assigned_label_name=dataset.class_names[req.label_id],
        annotator=annotator_tag,
        remaining_pending=counts["n_queued"],
        ready_to_retrain=(counts["n_queued"] == 0),
    )


def get_annotation_history(exp_id: str, limit: int = 200) -> AnnotationHistoryResponse:
    detail = get_experiment_detail(exp_id)
    summary = detail.summary
    cfg = detail.config

    dataset = get_cached_dataset(
        dataset_id=summary.dataset,
        max_samples=int(cfg.get("max_samples", 450)),
        use_synthetic=bool(cfg.get("use_synthetic", True)),
        seed=int(cfg.get("seed", 42)),
    )
    db = get_experiment_db(exp_id)
    # Only expose ground-truth comparison for samples already labeled (or simulated oracle runs)
    with db.get_session() as session:
        rows = (
            session.query(SampleRecord)
            .filter(SampleRecord.split == "pool", SampleRecord.status == "labeled")
            .order_by(SampleRecord.updated_at.desc())
            .limit(limit)
            .all()
        )
        total_count = (
            session.query(SampleRecord)
            .filter(SampleRecord.split == "pool", SampleRecord.status == "labeled")
            .count()
        )
        human_count = (
            session.query(SampleRecord)
            .filter(
                SampleRecord.split == "pool",
                SampleRecord.status == "labeled",
                SampleRecord.annotator.like("human%"),
            )
            .count()
        )
        oracle_count = (
            session.query(SampleRecord)
            .filter(
                SampleRecord.split == "pool",
                SampleRecord.status == "labeled",
                SampleRecord.annotator == "oracle",
            )
            .count()
        )

    items: List[AnnotationHistoryItem] = []
    for r in rows:
        ann = r.annotator or "unknown"
        if ann.startswith("human"):
            src_type = "human"
        elif ann == "oracle":
            src_type = "simulated_oracle"
        elif ann == "cold_start":
            src_type = "cold_start"
        else:
            src_type = "other"

        lbl = int(r.label) if r.label is not None else 0
        t_lbl = int(r.true_label) if r.true_label is not None else None
        items.append(
            AnnotationHistoryItem(
                sample_id=r.sample_id,
                experiment_id=exp_id,
                assigned_label_id=lbl,
                assigned_label_name=dataset.class_names[lbl],
                true_label_id=t_lbl,
                true_label_name=dataset.class_names[t_lbl] if t_lbl is not None else None,
                matches_ground_truth=(lbl == t_lbl) if t_lbl is not None else None,
                round_added=r.round_added,
                annotator=ann,
                source_type=src_type,
                updated_at=r.updated_at.isoformat() if r.updated_at else None,
            )
        )

    return AnnotationHistoryResponse(
        experiment_id=exp_id,
        total_count=total_count,
        human_count=human_count,
        oracle_count=oracle_count,
        items=items,
    )


def get_label_studio_status() -> LabelStudioStatusResponse:
    return LabelStudioStatusResponse(
        integration_status="pending_phase_3",
        label_studio_configured=bool(LABEL_STUDIO_URL),
        webhook_endpoint="/api/v1/webhooks/label-studio",
        supported_events=["ANNOTATION_CREATED", "ANNOTATION_UPDATED"],
        notes=(
            "Phase 3 integration pending: Label Studio SDK task sync and Cohen's Kappa QA are planned. "
            "The webhook receiver endpoint is ready to accept structured annotation payloads."
        ),
    )


def process_label_studio_webhook(payload: LabelStudioWebhookPayload) -> dict:
    """Validate and optionally ingest an external Label Studio annotation event."""
    if payload.experiment_id and payload.sample_id and payload.label_id is not None:
        submit_req = AnnotationSubmitRequest(
            experiment_id=payload.experiment_id,
            sample_id=payload.sample_id,
            label_id=payload.label_id,
            annotator=payload.annotator or "human_label_studio",
        )
        res = submit_annotation(submit_req)
        return {
            "accepted": True,
            "action": payload.action,
            "ingested": True,
            "result": res.model_dump(),
        }

    return {
        "accepted": True,
        "action": payload.action,
        "ingested": False,
        "message": "Webhook received. Provide experiment_id, sample_id, and label_id to ingest into SQLite.",
    }

