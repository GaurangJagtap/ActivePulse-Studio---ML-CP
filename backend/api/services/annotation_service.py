"""Human-in-the-Loop Annotation service, Label Studio SDK sync, webhooks, and QA."""

import hashlib
import hmac
import json
import os
from typing import Dict, List, Optional
import numpy as np

from backend.api.config import (
    AUTO_RETRAIN_ON_BATCH,
    BATCH_RETRAIN_THRESHOLD,
    DEFAULT_DOUBLE_ANNOTATION_RATE,
    LABEL_STUDIO_URL,
    LABEL_STUDIO_WEBHOOK_SECRET,
)
from backend.api.schemas.annotations import (
    AnnotationHistoryItem,
    AnnotationHistoryResponse,
    AnnotationQualityResponse,
    AnnotationSubmitRequest,
    AnnotationSubmitResponse,
    CompletedAnnotationItem,
    CompletedAnnotationsResponse,
    LabelStudioStatusResponse,
    LabelStudioSyncRequest,
    LabelStudioSyncResponse,
    LabelStudioTaskCreateRequest,
    LabelStudioTaskCreateResponse,
    LabelStudioWebhookPayload,
    LabelStudioWebhookResponse,
    PendingAnnotationSample,
    PendingAnnotationsResponse,
)
from backend.api.services.dataset_service import get_cached_dataset
from backend.api.services.experiment_service import (
    advance_experiment_round,
    get_experiment_db,
    get_experiment_detail,
    get_experiment_runtime,
    _append_log,
    _update_meta,
    _get_db_counts,
)
from backend.api.services.utils import feature_to_base64_png, summarize_text_embedding
from src.data.db import SampleRecord
from src.labeling.label_studio import (
    LabelStudioLabeler,
    LabelStudioManager,
    extract_label_from_ls_result,
)
from src.labeling.quality import evaluate_annotation_quality


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
        queued_records = (
            session.query(SampleRecord)
            .filter(
                SampleRecord.experiment_id == db.experiment_id,
                SampleRecord.split == "pool",
                SampleRecord.status == "queued",
            )
            .order_by(SampleRecord.sample_id.asc())
            .all()
        )
        queued_ids = [r.sample_id for r in queued_records]
        stored_pre = {
            r.sample_id: {
                "pred_id": r.pre_annotation_label,
                "confidence": r.pre_annotation_confidence,
                "probs": json.loads(r.pre_annotation_probs_json) if r.pre_annotation_probs_json else None,
            }
            for r in queued_records
        }

        human_count = (
            session.query(SampleRecord)
            .filter(
                SampleRecord.experiment_id == db.experiment_id,
                SampleRecord.split == "pool",
                SampleRecord.status == "labeled",
                SampleRecord.annotator.like("human%"),
            )
            .count()
        )
        total_labeled = (
            session.query(SampleRecord)
            .filter(
                SampleRecord.experiment_id == db.experiment_id,
                SampleRecord.split == "pool",
                SampleRecord.status == "labeled",
            )
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
            if p_vec is not None:
                pred_id = int(np.argmax(p_vec))
                conf = float(np.max(p_vec))
                prob_list = [round(float(v), 5) for v in p_vec]
            else:
                db_pre = stored_pre.get(sid, {})
                pred_id = db_pre.get("pred_id")
                conf = db_pre.get("confidence")
                prob_list = db_pre.get("probs")
            unc = (1.0 - float(conf)) if conf is not None else None

            samples.append(
                PendingAnnotationSample(
                    sample_id=sid,
                    experiment_id=exp_id,
                    dataset=dataset.name,
                    modality=modality,
                    round_num=summary.current_round,
                    predicted_class_id=pred_id,
                    predicted_class_name=dataset.class_names[pred_id] if pred_id is not None else None,
                    confidence=round(float(conf), 4) if conf is not None else None,
                    uncertainty=round(unc, 4) if unc is not None else None,
                    probabilities=prob_list,
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
    record = db.get_sample(req.sample_id)
    if record is None or record.split != "pool":
        raise KeyError(f"Pool sample '{req.sample_id}' not found in experiment '{req.experiment_id}'.")

    annotator_tag = req.annotator if req.annotator.startswith("human") else f"human_{req.annotator}"
    db.record_human_annotation(
        sample_id=req.sample_id,
        label_id=req.label_id,
        annotator=annotator_tag,
        round_num=summary.current_round,
        source_type="human",
        is_double_annotation=bool(record.needs_review and record.status == "labeled"),
    )

    # Continuous Learning feedback recording
    from backend.api.services.continuous_service import trigger_continuous_retraining_async

    orig_pred_id = record.pre_annotation_label
    orig_conf = record.pre_annotation_confidence
    runtime = get_experiment_runtime(req.experiment_id)
    if orig_pred_id is None and runtime is not None and getattr(runtime.orchestrator.model, "initialized_", False):
        try:
            feats = dataset.get_features([req.sample_id])
            probs = runtime.orchestrator.model.predict_calibrated_proba(feats)
            orig_pred_id = int(np.argmax(probs[0]))
            orig_conf = float(np.max(probs[0]))
        except Exception:
            pass

    orig_pred_name = dataset.class_names[orig_pred_id] if orig_pred_id is not None and 0 <= orig_pred_id < len(dataset.class_names) else None
    
    cl_feedback = db.record_continuous_feedback(
        sample_id=req.sample_id,
        human_label_id=req.label_id,
        human_label_name=dataset.class_names[req.label_id],
        annotator=annotator_tag,
        original_pred_id=orig_pred_id,
        original_pred_name=orig_pred_name,
        original_confidence=orig_conf,
    )
    is_correction = bool(cl_feedback["is_correction"])

    # Trigger automatic continuous retraining in background
    trigger_type = "auto_correction" if is_correction else "auto_confirmation"
    trigger_continuous_retraining_async(
        exp_id=req.experiment_id,
        trigger_type=trigger_type,
        trigger_sample_id=req.sample_id,
    )

    counts = _get_db_counts(db)
    _update_meta(
        req.experiment_id,
        n_labeled=counts["n_labeled"],
        n_pool=counts["n_pool"],
        n_queued=counts["n_queued"],
    )
    feedback_label_type = "CORRECTION" if is_correction else "CONFIRMATION"
    _append_log(
        req.experiment_id,
        f"Annotated sample {req.sample_id} as '{dataset.class_names[req.label_id]}' ({annotator_tag}) [{feedback_label_type}]. "
        f"Queued for continuous self-retraining. Pending queue: {counts['n_queued']}.",
    )

    return AnnotationSubmitResponse(
        experiment_id=req.experiment_id,
        sample_id=req.sample_id,
        assigned_label_id=req.label_id,
        assigned_label_name=dataset.class_names[req.label_id],
        annotator=annotator_tag,
        remaining_pending=counts["n_queued"],
        ready_to_retrain=(counts["n_queued"] == 0),
        is_correction=is_correction,
        continuous_learning_triggered=True,
        message=f"Human {feedback_label_type.lower()} saved. Model self-retraining initiated.",
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
            .filter(
                SampleRecord.experiment_id == db.experiment_id,
                SampleRecord.split == "pool",
                SampleRecord.status == "labeled",
            )
            .order_by(SampleRecord.updated_at.desc())
            .limit(limit)
            .all()
        )
        total_count = (
            session.query(SampleRecord)
            .filter(
                SampleRecord.experiment_id == db.experiment_id,
                SampleRecord.split == "pool",
                SampleRecord.status == "labeled",
            )
            .count()
        )
        human_count = (
            session.query(SampleRecord)
            .filter(
                SampleRecord.experiment_id == db.experiment_id,
                SampleRecord.split == "pool",
                SampleRecord.status == "labeled",
                SampleRecord.annotator.like("human%"),
            )
            .count()
        )
        oracle_count = (
            session.query(SampleRecord)
            .filter(
                SampleRecord.experiment_id == db.experiment_id,
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


def get_label_studio_status(manager: Optional[LabelStudioManager] = None) -> LabelStudioStatusResponse:
    ls_manager = manager or LabelStudioManager()
    status = ls_manager.health_status()
    return LabelStudioStatusResponse(
        integration_status=status["integration_status"],
        label_studio_configured=bool(status["configured"]),
        connected=bool(status["connected"]),
        base_url=status.get("base_url") or (LABEL_STUDIO_URL or None),
        default_project_id=status.get("default_project_id"),
        project_count=int(status.get("project_count", 0)),
        webhook_endpoint="/api/v1/webhooks/label-studio",
        supported_events=["ANNOTATION_CREATED", "ANNOTATION_UPDATED"],
        notes=(
            status["error"]
            if status.get("error")
            else "Label Studio SDK connected and ready for human-in-the-loop task sync and pre-annotations."
        ),
    )


def verify_webhook_secret(headers: Optional[Dict[str, str]], raw_body: bytes = b"") -> None:
    """Verify webhook shared secret or HMAC-SHA256 signature when configured."""
    secret = os.environ.get("LABEL_STUDIO_WEBHOOK_SECRET", "").strip() or LABEL_STUDIO_WEBHOOK_SECRET
    if not secret:
        return

    norm = {str(k).lower(): str(v).strip() for k, v in (headers or {}).items()}
    header_secret = norm.get("x-label-studio-secret", "")
    auth_header = norm.get("authorization", "")
    if header_secret and hmac.compare_digest(header_secret, secret):
        return
    if auth_header.lower().startswith("bearer "):
        token = auth_header[7:].strip()
        if hmac.compare_digest(token, secret):
            return

    sig_header = norm.get("x-label-studio-signature", "")
    if sig_header and raw_body:
        expected = hmac.new(secret.encode("utf-8"), raw_body, hashlib.sha256).hexdigest()
        candidate = sig_header.split("=", 1)[-1].strip()
        if hmac.compare_digest(candidate, expected):
            return

    raise PermissionError("Unauthorized Label Studio webhook: invalid or missing secret.")


def create_label_studio_tasks(
    req: LabelStudioTaskCreateRequest,
    manager: Optional[LabelStudioManager] = None,
) -> LabelStudioTaskCreateResponse:
    """Create Label Studio tasks with model pre-annotations for queued/requested samples."""
    detail = get_experiment_detail(req.experiment_id)
    summary = detail.summary
    cfg = detail.config

    dataset = get_cached_dataset(
        dataset_id=summary.dataset,
        max_samples=int(cfg.get("max_samples", 450)),
        use_synthetic=bool(cfg.get("use_synthetic", True)),
        seed=int(cfg.get("seed", 42)),
    )
    db = get_experiment_db(req.experiment_id)
    runtime = get_experiment_runtime(req.experiment_id)

    sample_ids = list(req.sample_ids or [])
    if not sample_ids:
        sample_ids = db.get_queued_sample_ids()
    if not sample_ids:
        raise ValueError(f"No queued or specified samples to upload for experiment '{req.experiment_id}'.")

    ls_manager = manager or LabelStudioManager()
    model_wrapper = (
        runtime.orchestrator.model
        if req.include_pre_annotations and runtime is not None
        else None
    )
    labeler = LabelStudioLabeler(
        db=db,
        dataset=dataset,
        project_id=req.project_id,
        manager=ls_manager,
        model_wrapper=model_wrapper,
        double_annotation_rate=req.double_annotation_rate or DEFAULT_DOUBLE_ANNOTATION_RATE,
    )
    batch_id = f"{req.experiment_id}_r{summary.current_round}"
    if req.webhook_url and labeler.project_id is None:
        modality = "image" if dataset.name == "cifar10" else "text"
        proj = ls_manager.ensure_project(
            title=f"ActivePulse - {dataset.name} ({req.experiment_id})",
            modality=modality,
            class_names=dataset.class_names,
            overlap_rate=labeler.double_annotation_rate,
            webhook_url=req.webhook_url,
        )
        labeler.project_id = int(proj["project_id"])

    result = labeler.queue_and_upload_batch(
        sample_ids=sample_ids,
        round_num=summary.current_round,
        strategy=summary.strategy,
        batch_id=batch_id,
        seed=int(cfg.get("seed", 42)) + summary.current_round,
    )
    _append_log(
        req.experiment_id,
        f"Uploaded {result['tasks_created']} sample(s) to Label Studio project {result['project_id']} "
        f"(double-annotation cohort={result['double_annotation_count']}).",
    )
    return LabelStudioTaskCreateResponse(
        experiment_id=req.experiment_id,
        project_id=int(result["project_id"]),
        tasks_created=int(result["tasks_created"]),
        double_annotation_count=int(result["double_annotation_count"]),
        batch_id=batch_id,
        mappings=result["mappings"],
    )


def sync_label_studio_annotations(
    req: LabelStudioSyncRequest,
    manager: Optional[LabelStudioManager] = None,
) -> LabelStudioSyncResponse:
    """Pull completed human annotations from Label Studio and store them idempotently."""
    detail = get_experiment_detail(req.experiment_id)
    summary = detail.summary
    cfg = detail.config

    dataset = get_cached_dataset(
        dataset_id=summary.dataset,
        max_samples=int(cfg.get("max_samples", 450)),
        use_synthetic=bool(cfg.get("use_synthetic", True)),
        seed=int(cfg.get("seed", 42)),
    )
    db = get_experiment_db(req.experiment_id)
    ls_manager = manager or LabelStudioManager()

    project_id = req.project_id or ls_manager.config.default_project_id
    if project_id is None:
        modality = "image" if dataset.name == "cifar10" else "text"
        proj = ls_manager.ensure_project(
            title=f"ActivePulse - {dataset.name} ({req.experiment_id})",
            modality=modality,
            class_names=dataset.class_names,
        )
        project_id = int(proj["project_id"])

    sync_res = ls_manager.sync_completed_annotations(
        project_id=int(project_id),
        class_names=dataset.class_names,
        db=db,
        round_num=summary.current_round,
        batch_id=req.batch_id or f"{req.experiment_id}_r{summary.current_round}",
    )

    counts = _get_db_counts(db)
    _update_meta(
        req.experiment_id,
        n_labeled=counts["n_labeled"],
        n_pool=counts["n_pool"],
        n_queued=counts["n_queued"],
    )
    if sync_res["synced_count"] > 0:
        _append_log(
            req.experiment_id,
            f"Synced {sync_res['synced_count']} completed human annotation(s) from Label Studio project {project_id}.",
        )

    return LabelStudioSyncResponse(
        experiment_id=req.experiment_id,
        project_id=int(project_id),
        synced_count=int(sync_res["synced_count"]),
        duplicate_skipped=int(sync_res["duplicate_skipped"]),
        invalid_skipped=int(sync_res["invalid_skipped"]),
        synced_sample_ids=list(sync_res["synced_sample_ids"]),
        ready_to_retrain=(counts["n_queued"] == 0),
    )


def process_label_studio_webhook(
    payload: LabelStudioWebhookPayload,
    headers: Optional[Dict[str, str]] = None,
    raw_body: bytes = b"",
) -> LabelStudioWebhookResponse:
    """Validate, deduplicate, and ingest a Label Studio annotation webhook event."""
    verify_webhook_secret(headers=headers, raw_body=raw_body)

    task = payload.task or {}
    ann = payload.annotation or {}
    task_meta = task.get("meta") if isinstance(task.get("meta"), dict) else {}
    task_data = task.get("data") if isinstance(task.get("data"), dict) else {}

    exp_id = payload.experiment_id or task_meta.get("experiment_id")
    if not exp_id:
        raise ValueError("Webhook payload must specify 'experiment_id' directly or inside task.meta.")

    detail = get_experiment_detail(str(exp_id))
    summary = detail.summary
    cfg = detail.config
    dataset = get_cached_dataset(
        dataset_id=summary.dataset,
        max_samples=int(cfg.get("max_samples", 450)),
        use_synthetic=bool(cfg.get("use_synthetic", True)),
        seed=int(cfg.get("seed", 42)),
    )
    db = get_experiment_db(str(exp_id))

    sample_id = payload.sample_id or task_meta.get("sample_id") or task_data.get("sample_id")
    ls_task_id = task.get("id")
    record = db.get_sample(str(sample_id)) if sample_id else None
    if record is None and ls_task_id is not None:
        record = db.get_sample_by_ls_task_id(int(ls_task_id))
        if record is not None:
            sample_id = record.sample_id

    if not sample_id or record is None:
        raise KeyError(f"Target sample '{sample_id or ls_task_id}' not found in experiment '{exp_id}'.")

    label_id = payload.label_id
    if label_id is None and ann:
        label_id = extract_label_from_ls_result(ann.get("result"), dataset.class_names)
    if label_id is None or label_id < 0 or label_id >= dataset.num_classes:
        raise ValueError(f"Invalid or missing annotation label for sample '{sample_id}'.")

    ann_id = ann.get("id") if isinstance(ann, dict) else None
    completed_by = ann.get("completed_by") if isinstance(ann, dict) else None
    if payload.annotator:
        annotator = payload.annotator
    elif isinstance(completed_by, dict):
        annotator = str(completed_by.get("email") or completed_by.get("username") or f"ls_user_{completed_by.get('id', 'human')}")
    elif completed_by is not None:
        annotator = f"ls_user_{completed_by}"
    else:
        annotator = "human_label_studio"
    if not annotator.startswith("human") and not annotator.startswith("ls_"):
        annotator = f"human_{annotator}"

    batch_id = task_meta.get("batch_id") or record.batch_id
    event_id = payload.webhook_event_id or (
        f"ls_webhook_{payload.action}_{ann_id}_{sample_id}" if ann_id is not None else None
    )

    inserted, resolved_batch_id = db.record_human_annotation(
        sample_id=str(sample_id),
        label_id=int(label_id),
        annotator=annotator,
        round_num=summary.current_round,
        source_type="label_studio",
        batch_id=batch_id,
        webhook_event_id=event_id,
        label_studio_annotation_id=int(ann_id) if ann_id is not None else None,
        is_double_annotation=bool(task_meta.get("double_annotation") or record.needs_review or record.status == "labeled"),
    )

    counts = _get_db_counts(db)
    _update_meta(
        str(exp_id),
        n_labeled=counts["n_labeled"],
        n_pool=counts["n_pool"],
        n_queued=counts["n_queued"],
    )

    batch_info = db.refresh_batch_progress(resolved_batch_id) if resolved_batch_id else None
    completed_cnt = batch_info["completed_tasks"] if batch_info else counts["n_labeled"]
    threshold_cnt = batch_info["threshold_count"] if batch_info else 0
    if batch_info and batch_info["total_tasks"] > 0:
        ready = (completed_cnt / batch_info["total_tasks"]) >= BATCH_RETRAIN_THRESHOLD and counts["n_queued"] == 0
    else:
        ready = counts["n_queued"] == 0

    retrain_triggered = False
    if inserted:
        _append_log(
            str(exp_id),
            f"Label Studio webhook ({payload.action}): sample {sample_id} -> "
            f"'{dataset.class_names[int(label_id)]}' by {annotator}.",
        )
        if AUTO_RETRAIN_ON_BATCH and ready and summary.mode == "human_in_the_loop" and summary.status != "running":
            try:
                advance_experiment_round(str(exp_id))
                retrain_triggered = True
            except Exception:
                retrain_triggered = False

    return LabelStudioWebhookResponse(
        status="duplicate_ignored" if not inserted else "ingested",
        action=payload.action,
        experiment_id=str(exp_id),
        sample_id=str(sample_id),
        assigned_label_id=int(label_id),
        duplicate_skipped=not inserted,
        batch_id=resolved_batch_id,
        batch_completed=int(completed_cnt),
        batch_threshold=int(threshold_cnt),
        ready_to_retrain=ready,
        retrain_triggered=retrain_triggered,
        message=(
            "Duplicate webhook event skipped idempotently."
            if not inserted
            else f"Annotation stored for '{sample_id}' ({dataset.class_names[int(label_id)]})."
        ),
    )


def get_annotation_quality(exp_id: str) -> AnnotationQualityResponse:
    """Compute inter-annotator Cohen's Kappa, class agreement, and disagreement flags."""
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
    raw_records = db.get_annotation_records()

    # Include only human / label_studio submissions (never synthetic oracle or ground truth)
    human_records = [
        r for r in raw_records if r.get("source_type") in ("human", "label_studio", "manual")
    ]
    report = evaluate_annotation_quality(
        annotations=human_records,
        class_names=dataset.class_names,
    )
    db.save_quality_metric(report, round_num=summary.current_round)

    return AnnotationQualityResponse(
        experiment_id=exp_id,
        total_annotations=report["total_annotations"],
        total_annotated_samples=report["total_annotated_samples"],
        overlapping_samples=report["overlapping_samples"],
        agreed_overlapping_samples=report["agreed_overlapping_samples"],
        disagreement_count=report["disagreement_count"],
        raw_agreement_rate=report["raw_agreement_rate"],
        cohens_kappa=report["cohens_kappa"],
        kappa_status_note=report["kappa_status_note"],
        pairwise_kappa=report["pairwise_kappa"],
        class_wise_agreement=report["class_wise_agreement"],
        disagreements=report["disagreements"],
        annotator_stats=report["annotator_stats"],
    )


def get_completed_annotations(exp_id: str, limit: int = 200) -> CompletedAnnotationsResponse:
    """Return individual annotation submissions recorded in `annotations` table."""
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
    rows = db.get_annotation_records()[-limit:]
    rows.reverse()

    items = [
        CompletedAnnotationItem(
            id=int(r["id"]),
            sample_id=str(r["sample_id"]),
            label_id=int(r["label_id"]),
            label_name=dataset.class_names[int(r["label_id"])]
            if 0 <= int(r["label_id"]) < len(dataset.class_names)
            else str(r["label_id"]),
            annotator=str(r["annotator"]),
            round_num=int(r["round_num"]),
            source_type=str(r["source_type"]),
            is_double_annotation=bool(r["is_double_annotation"]),
            batch_id=r.get("batch_id"),
            created_at=r.get("created_at"),
        )
        for r in rows
    ]
    return CompletedAnnotationsResponse(
        experiment_id=exp_id,
        total_completed=len(items),
        annotations=items,
    )

