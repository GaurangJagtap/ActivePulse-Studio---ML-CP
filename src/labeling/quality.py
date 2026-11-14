"""Annotation quality control: double-annotation selection, Cohen's kappa, and disagreement detection."""
from __future__ import annotations

import random
from collections import defaultdict
from itertools import combinations
from typing import Any, Dict, List, Optional, Sequence, Tuple


def select_double_annotation_samples(
    sample_ids: Sequence[str],
    overlap_rate: float = 0.10,
    seed: Optional[int] = None,
) -> List[str]:
    """Select a subset of sample IDs for blind double annotation.

    Args:
        sample_ids: Candidate sample IDs in the queried batch.
        overlap_rate: Fraction of samples to assign to two annotators (0.0 to 1.0).
        seed: Optional RNG seed for reproducible selection.

    Returns:
        List of selected sample IDs preserving input order.
    """
    if not sample_ids or overlap_rate <= 0.0:
        return []
    rate = min(1.0, max(0.0, float(overlap_rate)))
    n_select = max(1, int(round(len(sample_ids) * rate)))
    n_select = min(len(sample_ids), n_select)

    rng = random.Random(seed)
    chosen = set(rng.sample(list(sample_ids), k=n_select))
    return [sid for sid in sample_ids if sid in chosen]


def compute_cohens_kappa(
    labels_a: Sequence[int],
    labels_b: Sequence[int],
    min_pairs: int = 2,
) -> Tuple[Optional[float], Optional[str]]:
    """Compute Cohen's kappa between two human annotators' label lists.

    Never compares annotations against ground truth. Returns `(None, reason)`
    when insufficient overlap exists or class variance is degenerate.
    """
    if len(labels_a) != len(labels_b):
        raise ValueError("labels_a and labels_b must have identical length.")

    n = len(labels_a)
    if n < min_pairs:
        return None, f"Insufficient overlapping samples ({n} < {min_pairs} required)."

    classes = sorted(set(labels_a) | set(labels_b))
    counts_a: Dict[int, int] = defaultdict(int)
    counts_b: Dict[int, int] = defaultdict(int)
    agree = 0

    for a, b in zip(labels_a, labels_b):
        counts_a[a] += 1
        counts_b[b] += 1
        if a == b:
            agree += 1

    p_o = agree / n
    p_e = sum((counts_a[c] / n) * (counts_b[c] / n) for c in classes)

    denom = 1.0 - p_e
    if abs(denom) < 1e-12:
        if len(classes) <= 1:
            return (
                None,
                "All overlapping samples received the same class from both annotators; kappa is undefined.",
            )
        return None, "Expected agreement equals 1.0; kappa is undefined."

    kappa = (p_o - p_e) / denom
    return round(float(kappa), 4), None


def evaluate_annotation_quality(
    annotations: Sequence[Dict[str, Any]],
    class_names: Optional[Sequence[str]] = None,
    min_kappa_pairs: int = 2,
) -> Dict[str, Any]:
    """Compute inter-annotator agreement, Cohen's kappa, disagreements, and annotator stats.

    Only human / multi-annotator annotation submissions should be passed here.
    Distinct annotators per sample are paired strictly with each other (never ground truth).
    """
    # Map sample_id -> {annotator: latest_record}
    by_sample: Dict[str, Dict[str, Dict[str, Any]]] = defaultdict(dict)
    annotator_totals: Dict[str, int] = defaultdict(int)

    for rec in annotations:
        sid = str(rec["sample_id"])
        annotator = str(rec.get("annotator") or "unknown")
        by_sample[sid][annotator] = dict(rec)
        annotator_totals[annotator] += 1

    overlapping_samples: List[str] = []
    disagreements: List[Dict[str, Any]] = []
    n_agreed_overlap = 0

    # Pairwise labels accumulated per (annotator_1, annotator_2)
    pair_labels: Dict[Tuple[str, str], Tuple[List[int], List[int]]] = defaultdict(lambda: ([], []))
    pooled_a: List[int] = []
    pooled_b: List[int] = []

    class_overlap_total: Dict[int, int] = defaultdict(int)
    class_overlap_agree: Dict[int, int] = defaultdict(int)

    annotator_overlap_total: Dict[str, int] = defaultdict(int)
    annotator_overlap_agree: Dict[str, int] = defaultdict(int)

    for sid, ann_map in by_sample.items():
        if len(ann_map) < 2:
            continue

        overlapping_samples.append(sid)
        labels_by_ann = {ann: int(info["label_id"]) for ann, info in sorted(ann_map.items())}
        distinct_labels = set(labels_by_ann.values())
        unanimous = len(distinct_labels) == 1

        if unanimous:
            n_agreed_overlap += 1
        else:
            disagreements.append(
                {
                    "sample_id": sid,
                    "annotator_labels": labels_by_ann,
                    "annotator_label_names": (
                        {
                            ann: (class_names[lbl] if class_names and 0 <= lbl < len(class_names) else str(lbl))
                            for ann, lbl in labels_by_ann.items()
                        }
                    ),
                    "needs_review": True,
                }
            )

        # Class-wise tracking for overlapping classes involved in this sample
        for cls_id in distinct_labels:
            class_overlap_total[cls_id] += 1
            if unanimous:
                class_overlap_agree[cls_id] += 1

        # Compute majority/peer agreement per annotator on this overlapping sample
        label_freq: Dict[int, int] = defaultdict(int)
        for lbl in labels_by_ann.values():
            label_freq[lbl] += 1

        for ann, lbl in labels_by_ann.items():
            annotator_overlap_total[ann] += 1
            # Agrees with at least one other annotator on this sample
            if label_freq[lbl] >= 2:
                annotator_overlap_agree[ann] += 1

        # Accumulate ordered annotator pairs
        for ann1, ann2 in combinations(sorted(labels_by_ann.keys()), 2):
            lbl1 = labels_by_ann[ann1]
            lbl2 = labels_by_ann[ann2]
            pair_labels[(ann1, ann2)][0].append(lbl1)
            pair_labels[(ann1, ann2)][1].append(lbl2)
            pooled_a.append(lbl1)
            pooled_b.append(lbl2)

    n_overlap = len(overlapping_samples)
    raw_agreement_rate: Optional[float] = (
        round(n_agreed_overlap / n_overlap, 4) if n_overlap > 0 else None
    )

    # Overall kappa across pooled overlapping annotator pairs
    overall_kappa, kappa_reason = compute_cohens_kappa(pooled_a, pooled_b, min_pairs=min_kappa_pairs)

    pairwise_kappa: List[Dict[str, Any]] = []
    for (ann1, ann2), (la, lb) in sorted(pair_labels.items()):
        pk, reason = compute_cohens_kappa(la, lb, min_pairs=min_kappa_pairs)
        agree_cnt = sum(1 for a, b in zip(la, lb) if a == b)
        pairwise_kappa.append(
            {
                "annotator_a": ann1,
                "annotator_b": ann2,
                "n_overlapping": len(la),
                "agreement_rate": round(agree_cnt / len(la), 4) if la else None,
                "cohens_kappa": pk,
                "reason": reason,
            }
        )

    class_wise_agreement: List[Dict[str, Any]] = []
    for cls_id in sorted(class_overlap_total.keys()):
        total_c = class_overlap_total[cls_id]
        agree_c = class_overlap_agree[cls_id]
        cls_name = class_names[cls_id] if class_names and 0 <= cls_id < len(class_names) else str(cls_id)
        class_wise_agreement.append(
            {
                "class_id": cls_id,
                "class_name": cls_name,
                "overlapping_samples": total_c,
                "agreed_samples": agree_c,
                "agreement_rate": round(agree_c / total_c, 4) if total_c > 0 else None,
            }
        )

    annotator_stats: List[Dict[str, Any]] = []
    for ann in sorted(annotator_totals.keys()):
        ov_total = annotator_overlap_total[ann]
        ov_agree = annotator_overlap_agree[ann]
        annotator_stats.append(
            {
                "annotator": ann,
                "total_annotations": annotator_totals[ann],
                "overlapping_annotations": ov_total,
                "peer_agreement_rate": round(ov_agree / ov_total, 4) if ov_total > 0 else None,
            }
        )

    return {
        "total_annotations": len(annotations),
        "total_annotated_samples": len(by_sample),
        "overlapping_samples": n_overlap,
        "overlapping_sample_ids": overlapping_samples,
        "agreed_overlapping_samples": n_agreed_overlap,
        "disagreement_count": len(disagreements),
        "raw_agreement_rate": raw_agreement_rate,
        "cohens_kappa": overall_kappa,
        "kappa_status_note": kappa_reason or "Computed from independent human annotator pairs.",
        "pairwise_kappa": pairwise_kappa,
        "class_wise_agreement": class_wise_agreement,
        "disagreements": disagreements,
        "annotator_stats": annotator_stats,
    }

