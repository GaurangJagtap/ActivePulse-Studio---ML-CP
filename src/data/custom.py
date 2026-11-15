"""Custom dataset adapter supporting uploaded image folders (ZIP) and tabular/text CSVs."""

import json
import os
import shutil
import zipfile
from typing import Dict, List, Optional, Tuple
import numpy as np
from PIL import Image

from src.data.base import DatasetAdapter


class CustomDatasetAdapter(DatasetAdapter):
    """Adapter for user-uploaded custom datasets (Images from ZIP or Text/Tabular from CSV)."""

    def __init__(
        self,
        dataset_id: str,
        upload_dir: str,
        max_samples: Optional[int] = None,
        seed: int = 42,
    ):
        self.dataset_id = dataset_id.lower().strip()
        self.upload_dir = upload_dir
        self.max_samples = max_samples
        self.seed = seed

        self._sample_ids: List[str] = []
        self._class_names: List[str] = []
        self._data_map: Dict[str, Tuple[np.ndarray, int, Optional[str]]] = {}  # id -> (feature, label, raw_text_or_path)
        self._modality: str = "image"
        self._metadata: Dict = {}

        self._load_custom_dataset()

    @property
    def name(self) -> str:
        return self.dataset_id

    @property
    def modality(self) -> str:
        return self._modality

    @property
    def num_classes(self) -> int:
        return max(1, len(self._class_names))

    @property
    def class_names(self) -> List[str]:
        return list(self._class_names)

    def _load_custom_dataset(self) -> None:
        meta_file = os.path.join(self.upload_dir, "dataset_meta.json")
        if not os.path.exists(meta_file):
            raise FileNotFoundError(f"Custom dataset directory '{self.upload_dir}' missing dataset_meta.json.")

        with open(meta_file, "r", encoding="utf-8") as f:
            self._metadata = json.load(f)

        self._modality = self._metadata.get("modality", "image")
        self._class_names = self._metadata.get("class_names", ["unlabeled"])

        data_bin = os.path.join(self.upload_dir, "data.npz")
        if not os.path.exists(data_bin):
            raise FileNotFoundError(f"Missing data.npz in {self.upload_dir}")

        archive = np.load(data_bin, allow_pickle=True)
        all_ids = [str(x) for x in archive["sample_ids"]]
        features = archive["features"]
        labels = archive["labels"]
        raw_texts = archive["raw_texts"] if "raw_texts" in archive else None

        total = len(all_ids)
        if self.max_samples and self.max_samples < total:
            rng = np.random.RandomState(self.seed)
            indices = np.sort(rng.choice(total, size=self.max_samples, replace=False))
            all_ids = [all_ids[i] for i in indices]
            features = features[indices]
            labels = labels[indices]
            if raw_texts is not None:
                raw_texts = raw_texts[indices]

        self._sample_ids = all_ids
        for i, sid in enumerate(all_ids):
            txt = str(raw_texts[i]) if raw_texts is not None else None
            self._data_map[sid] = (features[i], int(labels[i]), txt)

    def get_all_sample_ids(self) -> List[str]:
        return list(self._sample_ids)

    def get_features(self, sample_ids: List[str]) -> np.ndarray:
        feats = [self._data_map[sid][0] for sid in sample_ids]
        return np.stack(feats, axis=0).astype(np.float32)

    def get_ground_truth(self, sample_ids: List[str]) -> np.ndarray:
        return np.array([self._data_map[sid][1] for sid in sample_ids], dtype=np.int64)

    def get_raw_text(self, sample_id: str) -> Optional[str]:
        if sample_id in self._data_map:
            return self._data_map[sample_id][2]
        return None
