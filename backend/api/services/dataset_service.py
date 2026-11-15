"""Dataset inspection service querying DatasetAdapter and PoolManager."""

import threading
from typing import Dict, List, Optional, Tuple
import numpy as np

from backend.api.schemas.common import (
    ClassCount,
    DatasetSamplePreview,
    DatasetSamplesResponse,
    DatasetStatsResponse,
    SplitStats,
)
from backend.api.services.utils import (
    build_dataset_adapter,
    feature_to_base64_png,
    summarize_text_embedding,
)
from src.data.base import DatasetAdapter
from src.data.db import SampleRecord


_DATASET_CACHE: Dict[Tuple[str, int, bool, int], DatasetAdapter] = {}
_CACHE_LOCK = threading.Lock()


def get_cached_dataset(
    dataset_id: str,
    max_samples: int = 450,
    use_synthetic: bool = True,
    seed: int = 42,
) -> DatasetAdapter:
    key = (dataset_id.lower(), max_samples, bool(use_synthetic), seed)
    with _CACHE_LOCK:
        if key not in _DATASET_CACHE:
            _DATASET_CACHE[key] = build_dataset_adapter(
                dataset_id=dataset_id,
                max_samples=max_samples,
                use_synthetic=use_synthetic,
                seed=seed,
            )
        return _DATASET_CACHE[key]


def _deterministic_split_preview(n_total: int, test_ratio: float = 0.2, val_ratio: float = 0.1, seed: int = 42):
    rng = np.random.RandomState(seed)
    indices = np.arange(n_total)
    rng.shuffle(indices)
    n_test = int(n_total * test_ratio)
    n_val = int(n_total * val_ratio)
    test_idx = set(indices[:n_test])
    val_idx = set(indices[n_test : n_test + n_val])
    splits = []
    for i in range(n_total):
        if i in test_idx:
            splits.append("test")
        elif i in val_idx:
            splits.append("val")
        else:
            splits.append("pool")
    return splits


def get_dataset_stats(
    dataset_id: str,
    max_samples: int = 450,
    use_synthetic: bool = True,
    seed: int = 42,
    experiment_db=None,
) -> DatasetStatsResponse:
    adapter = get_cached_dataset(dataset_id, max_samples=max_samples, use_synthetic=use_synthetic, seed=seed)
    all_ids = adapter.get_all_sample_ids()
    ground_truth = adapter.get_ground_truth(all_ids)

    counts = np.bincount(ground_truth, minlength=adapter.num_classes)
    class_dist = [
        ClassCount(
            class_id=i,
            class_name=adapter.class_names[i],
            count=int(counts[i]),
        )
        for i in range(adapter.num_classes)
    ]

    sample_feat = adapter.get_features(all_ids[:1])
    feature_shape = list(sample_feat.shape[1:])

    if experiment_db is not None:
        with experiment_db.get_session() as session:
            rows = session.query(SampleRecord.split, SampleRecord.status).all()
            n_total = len(rows)
            n_pool_unlabeled = sum(1 for sp, st in rows if sp == "pool" and st == "unlabeled")
            n_queued = sum(1 for sp, st in rows if sp == "pool" and st == "queued")
            n_labeled = sum(1 for sp, st in rows if sp == "pool" and st == "labeled")
            n_val = sum(1 for sp, _ in rows if sp == "val")
            n_test = sum(1 for sp, _ in rows if sp == "test")
    else:
        splits = _deterministic_split_preview(len(all_ids), seed=seed)
        n_total = len(all_ids)
        n_test = sum(1 for s in splits if s == "test")
        n_val = sum(1 for s in splits if s == "val")
        n_pool_unlabeled = sum(1 for s in splits if s == "pool")
        n_queued = 0
        n_labeled = 0

    modality = getattr(adapter, "modality", "image" if adapter.name == "cifar10" else "text_embedding")
    is_custom = adapter.name.startswith("custom_")
    display_name = (
        f"{adapter.name.replace('custom_', '').title()} (Custom Upload)"
        if is_custom
        else ("CIFAR-10 (Image Vision)" if adapter.name == "cifar10" else "AG News (Text Embeddings)")
    )
    return DatasetStatsResponse(
        dataset_id=adapter.name,
        name=display_name,
        modality=modality,
        use_synthetic=use_synthetic,
        max_samples=len(all_ids),
        num_classes=adapter.num_classes,
        class_names=adapter.class_names,
        upload_supported=True,
        upload_status_note="Custom image (ZIP) and text (CSV) upload is active. Uploaded datasets can be queried and labeled in real-time.",
        split_stats=SplitStats(
            n_total=n_total,
            n_pool_unlabeled=n_pool_unlabeled,
            n_queued=n_queued,
            n_labeled=n_labeled,
            n_val=n_val,
            n_test=n_test,
        ),
        class_distribution=class_dist,
        feature_shape=feature_shape,
    )


def get_dataset_samples(
    dataset_id: str,
    offset: int = 0,
    limit: int = 24,
    max_samples: int = 450,
    use_synthetic: bool = True,
    seed: int = 42,
    split_filter: Optional[str] = None,
    status_filter: Optional[str] = None,
    experiment_db=None,
) -> DatasetSamplesResponse:
    adapter = get_cached_dataset(dataset_id, max_samples=max_samples, use_synthetic=use_synthetic, seed=seed)
    all_ids = adapter.get_all_sample_ids()
    gt_all = adapter.get_ground_truth(all_ids)

    records_meta: List[Dict] = []
    if experiment_db is not None:
        with experiment_db.get_session() as session:
            query = session.query(SampleRecord).order_by(SampleRecord.sample_id.asc())
            if split_filter:
                query = query.filter(SampleRecord.split == split_filter)
            if status_filter:
                query = query.filter(SampleRecord.status == status_filter)
            db_rows = query.all()
            for r in db_rows:
                records_meta.append(
                    {
                        "sample_id": r.sample_id,
                        "split": r.split,
                        "status": r.status,
                        "true_label": r.true_label,
                        "label": r.label,
                        "annotator": r.annotator,
                    }
                )
    else:
        splits = _deterministic_split_preview(len(all_ids), seed=seed)
        for sid, sp, t_lbl in zip(all_ids, splits, gt_all):
            status = "unlabeled" if sp == "pool" else "labeled"
            if split_filter and sp != split_filter:
                continue
            if status_filter and status != status_filter:
                continue
            records_meta.append(
                {
                    "sample_id": sid,
                    "split": sp,
                    "status": status,
                    "true_label": int(t_lbl),
                    "label": int(t_lbl) if sp != "pool" else None,
                    "annotator": "pre_split" if sp != "pool" else None,
                }
            )

    total = len(records_meta)
    page = records_meta[offset : offset + limit]
    if not page:
        return DatasetSamplesResponse(
            dataset_id=adapter.name,
            total=total,
            offset=offset,
            limit=limit,
            samples=[],
        )

    page_ids = [item["sample_id"] for item in page]
    feats = adapter.get_features(page_ids)

    samples: List[DatasetSamplePreview] = []
    is_image = getattr(adapter, "modality", "image" if adapter.name == "cifar10" else "text") == "image"
    for i, item in enumerate(page):
        feat = feats[i]
        t_lbl = item["true_label"]
        a_lbl = item["label"]
        img_b64 = feature_to_base64_png(feat) if is_image else None
        feat_summary = summarize_text_embedding(feat) if not is_image else None
        emb_preview = [round(float(v), 4) for v in feat.reshape(-1)[:16]] if not is_image else None

        samples.append(
            DatasetSamplePreview(
                sample_id=item["sample_id"],
                split=item["split"],
                status=item["status"],
                true_class_id=t_lbl,
                true_class_name=adapter.class_names[t_lbl] if t_lbl is not None else None,
                assigned_class_name=adapter.class_names[a_lbl] if a_lbl is not None else None,
                annotator=item["annotator"],
                image_base64=img_b64,
                feature_summary=feat_summary,
                embedding_preview=emb_preview,
            )
        )

    return DatasetSamplesResponse(
        dataset_id=adapter.name,
        total=total,
        offset=offset,
        limit=limit,
        samples=samples,
    )

