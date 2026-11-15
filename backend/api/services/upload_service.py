"""Service for uploading, parsing, and storing custom datasets (ZIP images or CSV text)."""

import json
import os
import re
import shutil
import zipfile
from typing import Dict, List, Optional, Tuple
import numpy as np
from PIL import Image

from backend.api.config import DATA_DIR
from backend.api.schemas.common import DatasetSummary
from src.data.custom import CustomDatasetAdapter


def _custom_datasets_root() -> str:
    root = os.path.join(DATA_DIR, "custom_datasets")
    os.makedirs(root, exist_ok=True)
    return root


def _sanitize_dataset_id(name: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_\-]", "_", name).lower().strip("_-")
    if not cleaned:
        cleaned = "custom_dataset"
    return f"custom_{cleaned}"


def list_custom_dataset_summaries() -> List[DatasetSummary]:
    """Scan custom_datasets folder and return DatasetSummary objects."""
    root = _custom_datasets_root()
    summaries = []
    if not os.path.exists(root):
        return summaries

    for item in sorted(os.listdir(root)):
        ds_dir = os.path.join(root, item)
        if not os.path.isdir(ds_dir):
            continue
        meta_file = os.path.join(ds_dir, "dataset_meta.json")
        if not os.path.exists(meta_file):
            continue
        try:
            with open(meta_file, "r", encoding="utf-8") as f:
                meta = json.load(f)
            modality = meta.get("modality", "image")
            supported_models = ["cifar_cnn", "resnet18"] if modality == "image" else ["text_mlp"]
            default_model = supported_models[0]
            summary = DatasetSummary(
                id=meta["id"],
                name=f"{meta['name']} (Custom Upload)",
                modality=modality,
                num_classes=len(meta.get("class_names", [])),
                class_names=meta.get("class_names", []),
                default_model=default_model,
                supported_models=supported_models,
                supports_real_download=False,
                supports_raw_text=(modality == "text"),
                description=meta.get(
                    "description",
                    f"User uploaded dataset ({meta.get('total_samples', 0)} samples, {modality})."
                ),
            )
            summaries.append(summary)
        except Exception:
            continue
    return summaries


def get_custom_dataset_dir(dataset_id: str) -> Optional[str]:
    ds_dir = os.path.join(_custom_datasets_root(), dataset_id)
    if os.path.exists(os.path.join(ds_dir, "dataset_meta.json")):
        return ds_dir
    return None


def process_image_zip_upload(
    zip_bytes: bytes,
    dataset_name: str,
    target_size: Tuple[int, int] = (32, 32),
    max_samples: int = 2000,
) -> Dict:
    """Process a ZIP file containing image files (organized in folders or flat)."""
    dataset_id = _sanitize_dataset_id(dataset_name)
    ds_dir = os.path.join(_custom_datasets_root(), dataset_id)
    os.makedirs(ds_dir, exist_ok=True)

    temp_extract = os.path.join(ds_dir, "_temp_extract")
    os.makedirs(temp_extract, exist_ok=True)

    temp_zip = os.path.join(ds_dir, "_upload.zip")
    with open(temp_zip, "wb") as f:
        f.write(zip_bytes)

    with zipfile.ZipFile(temp_zip, "r") as zf:
        zf.extractall(temp_extract)

    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    image_paths: List[str] = []
    class_labels: List[str] = []

    for root, _, files in os.walk(temp_extract):
        for f in files:
            ext = os.path.splitext(f)[1].lower()
            if ext in valid_exts and not f.startswith("."):
                full_path = os.path.join(root, f)
                rel_dir = os.path.relpath(root, temp_extract)
                class_name = rel_dir.replace("\\", "/").split("/")[0] if rel_dir != "." else "unlabeled"
                image_paths.append(full_path)
                class_labels.append(class_name)

    if not image_paths:
        shutil.rmtree(ds_dir, ignore_errors=True)
        raise ValueError("No valid image files (.png, .jpg, .jpeg, .bmp, .webp) found in the ZIP archive.")

    unique_classes = sorted(list(set(class_labels)))
    class_to_idx = {c: idx for idx, c in enumerate(unique_classes)}

    if len(image_paths) > max_samples:
        indices = np.random.RandomState(42).choice(len(image_paths), size=max_samples, replace=False)
        image_paths = [image_paths[i] for i in indices]
        class_labels = [class_labels[i] for i in indices]

    processed_tensors = []
    sample_ids = []
    int_labels = []

    # Standard CIFAR-style mean/std for normalization
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)

    for i, (p, c_name) in enumerate(zip(image_paths, class_labels)):
        try:
            with Image.open(p) as img:
                img_rgb = img.convert("RGB").resize(target_size, Image.Resampling.BILINEAR)
                arr = np.array(img_rgb, dtype=np.float32) / 255.0  # (32, 32, 3)
                arr = np.transpose(arr, (2, 0, 1))  # (3, 32, 32)
                arr = (arr - mean) / std
                processed_tensors.append(arr)
                sid = f"{dataset_id}_{i:06d}"
                sample_ids.append(sid)
                int_labels.append(class_to_idx[c_name])
        except Exception:
            continue

    if not processed_tensors:
        shutil.rmtree(ds_dir, ignore_errors=True)
        raise ValueError("Could not decode any images from the uploaded archive.")

    features = np.stack(processed_tensors, axis=0).astype(np.float32)
    labels = np.array(int_labels, dtype=np.int64)

    # Save to data.npz
    npz_path = os.path.join(ds_dir, "data.npz")
    np.savez_compressed(
        npz_path,
        sample_ids=np.array(sample_ids),
        features=features,
        labels=labels,
    )

    # Save dataset_meta.json
    meta = {
        "id": dataset_id,
        "name": dataset_name,
        "modality": "image",
        "total_samples": len(sample_ids),
        "class_names": unique_classes,
        "num_classes": len(unique_classes),
        "target_size": list(target_size),
        "description": f"Custom uploaded image dataset: {len(sample_ids)} images across {len(unique_classes)} class(es).",
    }
    with open(os.path.join(ds_dir, "dataset_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    # Cleanup temp extraction
    shutil.rmtree(temp_extract, ignore_errors=True)
    if os.path.exists(temp_zip):
        os.remove(temp_zip)

    return meta


def process_csv_text_upload(
    csv_bytes: bytes,
    dataset_name: str,
    text_column: str = "text",
    label_column: Optional[str] = "label",
    max_samples: int = 2000,
) -> Dict:
    """Process a CSV file containing text and optional labels into dense embeddings."""
    import io
    import pandas as pd
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import TruncatedSVD

    dataset_id = _sanitize_dataset_id(dataset_name)
    ds_dir = os.path.join(_custom_datasets_root(), dataset_id)
    os.makedirs(ds_dir, exist_ok=True)

    df = pd.read_csv(io.BytesIO(csv_bytes))
    if text_column not in df.columns:
        # Auto-detect first string column
        text_cols = [c for c in df.columns if df[c].dtype == object]
        if not text_cols:
            shutil.rmtree(ds_dir, ignore_errors=True)
            raise ValueError(f"No text column found in CSV. Expected '{text_column}' or text column.")
        text_column = text_cols[0]

    df = df.dropna(subset=[text_column])
    if len(df) > max_samples:
        df = df.sample(n=max_samples, random_state=42).reset_index(drop=True)

    texts = df[text_column].astype(str).tolist()

    if label_column and label_column in df.columns:
        raw_labels = df[label_column].astype(str).tolist()
        unique_classes = sorted(list(set(raw_labels)))
        class_to_idx = {c: i for i, c in enumerate(unique_classes)}
        labels = np.array([class_to_idx[c] for c in raw_labels], dtype=np.int64)
    else:
        unique_classes = ["general", "topic_a", "topic_b"]
        labels = np.random.RandomState(42).randint(0, len(unique_classes), size=len(texts)).astype(np.int64)

    # Transform text to 64-dimensional dense representation
    tfidf = TfidfVectorizer(max_features=500, stop_words="english")
    tfidf_mat = tfidf.fit_transform(texts)
    n_dim = min(64, tfidf_mat.shape[1] - 1 if tfidf_mat.shape[1] > 2 else 2)
    svd = TruncatedSVD(n_components=n_dim, random_state=42)
    embeddings = svd.fit_transform(tfidf_mat)
    # Pad to exactly 64 dims if necessary
    if embeddings.shape[1] < 64:
        pad = np.zeros((embeddings.shape[0], 64 - embeddings.shape[1]), dtype=np.float32)
        embeddings = np.hstack([embeddings, pad])

    features = embeddings.astype(np.float32)
    sample_ids = [f"{dataset_id}_{i:06d}" for i in range(len(texts))]

    npz_path = os.path.join(ds_dir, "data.npz")
    np.savez_compressed(
        npz_path,
        sample_ids=np.array(sample_ids),
        features=features,
        labels=labels,
        raw_texts=np.array(texts),
    )

    meta = {
        "id": dataset_id,
        "name": dataset_name,
        "modality": "text",
        "total_samples": len(sample_ids),
        "class_names": unique_classes,
        "num_classes": len(unique_classes),
        "description": f"Custom uploaded text dataset: {len(sample_ids)} texts across {len(unique_classes)} classes.",
    }
    with open(os.path.join(ds_dir, "dataset_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    return meta
