"""Reusable ML-UI bridge utilities (extracted from legacy UI & centralized for FastAPI).

Includes dataset/model factory helpers, image tensor-to-PNG base64 serialization,
2D PCA latent-space projection, and annotation cost-savings estimation.
"""

import base64
import io
from typing import Dict, List, Optional, Tuple
import numpy as np
from PIL import Image
from sklearn.decomposition import PCA

from src.data.agnews import AGNewsAdapter
from src.data.base import DatasetAdapter
from src.data.cifar10 import CIFAR10Adapter
from src.models.cnn import CIFARConvNet, ResNet18TransferNet
from src.models.text_net import TextMLP
from src.models.wrapper import ActiveLearningClassifier


def build_dataset_adapter(
    dataset_id: str,
    max_samples: int = 450,
    use_synthetic: bool = True,
    seed: int = 42,
) -> DatasetAdapter:
    """Instantiate the appropriate DatasetAdapter from src/data/ or custom uploads."""
    key = dataset_id.lower()
    if key == "cifar10":
        return CIFAR10Adapter(
            max_samples=max_samples,
            use_synthetic=use_synthetic,
            seed=seed,
        )
    if key == "agnews":
        return AGNewsAdapter(
            max_samples=max_samples,
            use_synthetic=use_synthetic,
            seed=seed,
        )

    # Check for custom uploaded dataset
    from backend.api.services.upload_service import get_custom_dataset_dir
    from src.data.custom import CustomDatasetAdapter
    custom_dir = get_custom_dataset_dir(dataset_id)
    if custom_dir:
        return CustomDatasetAdapter(
            dataset_id=dataset_id,
            upload_dir=custom_dir,
            max_samples=max_samples,
            seed=seed,
        )

    raise ValueError(f"Unsupported dataset '{dataset_id}'. Supported: ['cifar10', 'agnews'] or custom uploaded datasets.")


def build_module_factory(dataset: DatasetAdapter, model_type: str):
    """Return a PyTorch nn.Module factory compatible with skorch NeuralNetClassifier."""
    mtype = model_type.lower()
    ds_name = dataset.name.lower()
    modality = getattr(dataset, "modality", "image")

    if mtype == "cifar_cnn":
        return lambda: CIFARConvNet(num_classes=dataset.num_classes, dropout_rate=0.2)

    if mtype == "resnet18":
        return lambda: ResNet18TransferNet(
            num_classes=dataset.num_classes, pretrained=False, dropout_rate=0.2
        )

    if mtype == "text_mlp":
        in_feat = 64
        return lambda: TextMLP(in_features=in_feat, num_classes=dataset.num_classes, dropout_rate=0.2)

    raise ValueError(
        f"Unsupported model_type '{model_type}'. Supported: ['cifar_cnn', 'resnet18', 'text_mlp']."
    )


def build_classifier(
    dataset: DatasetAdapter,
    model_type: str,
    max_epochs: int = 6,
    lr: float = 0.01,
    warm_start: bool = False,
    device: str = "cpu",
    batch_size: int = 16,
) -> ActiveLearningClassifier:
    """Create an ActiveLearningClassifier wrapping a PyTorch module."""
    module_cls = build_module_factory(dataset, model_type)
    return ActiveLearningClassifier(
        module=module_cls,
        max_epochs=max_epochs,
        lr=lr,
        warm_start=warm_start,
        device=device,
        batch_size=batch_size,
        verbose=0,
    )


def feature_to_base64_png(feat: np.ndarray) -> Optional[str]:
    """Convert a normalized (3, H, W) image tensor into a base64 PNG data URL."""
    if feat.ndim != 3 or feat.shape[0] != 3:
        return None
    img = np.transpose(feat, (1, 2, 0))
    denom = float(img.max() - img.min()) + 1e-6
    img_u8 = np.clip((img - img.min()) / denom * 255.0, 0, 255).astype(np.uint8)
    pil_img = Image.fromarray(img_u8)
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    encoded = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def summarize_text_embedding(feat: np.ndarray, top_k: int = 8) -> Dict[str, float]:
    """Summarize a 1D dense text embedding vector for UI inspection."""
    flat = feat.reshape(-1).astype(float)
    norm = float(np.linalg.norm(flat))
    mean = float(np.mean(flat))
    std = float(np.std(flat))
    max_val = float(np.max(flat))
    min_val = float(np.min(flat))
    top_idx = np.argsort(np.abs(flat))[::-1][:top_k]
    summary: Dict[str, float] = {
        "l2_norm": round(norm, 4),
        "mean": round(mean, 4),
        "std": round(std, 4),
        "min": round(min_val, 4),
        "max": round(max_val, 4),
    }
    for idx in top_idx:
        summary[f"dim_{int(idx)}"] = round(float(flat[idx]), 4)
    return summary


def compute_pca_projection(
    dataset: DatasetAdapter,
    model: Optional[ActiveLearningClassifier],
    labeled_ids: List[str],
    newly_queried_ids: List[str],
    max_points: int = 200,
) -> Tuple[List[Dict], List[float]]:
    """Project bottleneck embeddings (or raw features) to 2D via PCA."""
    all_ids = dataset.get_all_sample_ids()
    # Ensure labeled and newly queried points are included in the visualization window
    priority_ids: List[str] = []
    seen = set()
    for sid in newly_queried_ids + labeled_ids + all_ids:
        if sid not in seen:
            priority_ids.append(sid)
            seen.add(sid)
        if len(priority_ids) >= max_points:
            break

    feats = dataset.get_features(priority_ids)
    if model is not None and getattr(model, "initialized_", False):
        embeds = model.extract_embeddings(feats)
    else:
        embeds = feats.reshape(len(priority_ids), -1)

    if len(priority_ids) < 2 or embeds.shape[1] < 2:
        return [], [0.0, 0.0]

    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(embeds)
    explained = [round(float(v), 4) for v in pca.explained_variance_ratio_]

    labeled_set = set(labeled_ids)
    queried_set = set(newly_queried_ids)

    points = []
    for i, sid in enumerate(priority_ids):
        if sid in queried_set:
            category = "Newly Queried by Active Learning"
        elif sid in labeled_set:
            category = "Already Labeled"
        else:
            category = "Unlabeled Pool"

        points.append(
            {
                "sample_id": sid,
                "pc1": round(float(coords[i, 0]), 5),
                "pc2": round(float(coords[i, 1]), 5),
                "category": category,
            }
        )

    return points, explained


def estimate_cost_savings(total_samples: int, n_labeled: int) -> Dict:
    """Compute annotation cost savings estimate consistent with platform spec.

    Uses 75% passive random baseline assumption, 3.5 minutes/label, $32/hour annotator rate.
    """
    if total_samples <= 0:
        return {
            "total_samples": 0,
            "n_labeled": 0,
            "pct_dataset_labeled": 0.0,
            "estimated_passive_labels": 0,
            "estimated_labels_saved": 0,
            "estimated_hours_saved": 0.0,
            "estimated_dollars_saved": 0.0,
            "methodology_note": "No samples recorded.",
        }

    pct_labeled = (n_labeled / total_samples) * 100.0
    passive_equiv = int(total_samples * 0.75)
    labels_saved_est = max(0, passive_equiv - int(n_labeled))
    hours_saved = (labels_saved_est * 3.5) / 60.0
    dollars_saved = hours_saved * 32.0

    return {
        "total_samples": int(total_samples),
        "n_labeled": int(n_labeled),
        "pct_dataset_labeled": round(pct_labeled, 2),
        "estimated_passive_labels": passive_equiv,
        "estimated_labels_saved": labels_saved_est,
        "estimated_hours_saved": round(hours_saved, 2),
        "estimated_dollars_saved": round(dollars_saved, 2),
        "methodology_note": (
            "Estimated against a 75% passive labeling baseline at 3.5 min/sample and $32/hr expert rate."
        ),
    }

