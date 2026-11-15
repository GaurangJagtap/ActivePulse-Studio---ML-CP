"""Catalog service exposing verified datasets, PyTorch models, and query strategies."""

from typing import List
from backend.api.schemas.common import DatasetSummary, ModelInfo, StrategyInfo
from src.data.agnews import AGNewsAdapter
from src.data.cifar10 import CIFAR10Adapter
from src.strategies.registry import list_registered_strategies


_DATASET_CATALOG: List[DatasetSummary] = [
    DatasetSummary(
        id="cifar10",
        name="CIFAR-10 (Image Vision)",
        modality="image",
        num_classes=10,
        class_names=CIFAR10Adapter.CLASS_NAMES,
        default_model="cifar_cnn",
        supported_models=["cifar_cnn", "resnet18"],
        supports_real_download=True,
        supports_raw_text=False,
        description=(
            "32x32 RGB object classification across 10 classes. Supports both synthetic fast-CPU mode "
            "and torchvision CIFAR-10 download."
        ),
    ),
    DatasetSummary(
        id="agnews",
        name="AG News (Text Embeddings)",
        modality="text_embedding",
        num_classes=4,
        class_names=AGNewsAdapter.CLASS_NAMES,
        default_model="text_mlp",
        supported_models=["text_mlp"],
        supports_real_download=False,
        supports_raw_text=False,
        description=(
            "4-class news topic classification (World, Sports, Business, Sci/Tech). "
            "Current backend adapter generates 64-dimensional dense text embeddings."
        ),
    ),
]


_MODEL_CATALOG: List[ModelInfo] = [
    ModelInfo(
        id="cifar_cnn",
        name="CIFARConvNet (4-Layer CNN)",
        architecture="Conv2d + BatchNorm + MaxPool + 128-d Bottleneck + MC-Dropout",
        supported_datasets=["cifar10"],
        embedding_dim=128,
        supports_mc_dropout=True,
        description=(
            "Custom 32x32 convolutional network exposing 128-d bottleneck features and spatial dropout "
            "for BALD, Core-Set, BADGE, and Hybrid strategies."
        ),
    ),
    ModelInfo(
        id="resnet18",
        name="ResNet18TransferNet",
        architecture="ResNet-18 Backbone + Dropout + Linear Classifier",
        supported_datasets=["cifar10"],
        embedding_dim=512,
        supports_mc_dropout=True,
        description=(
            "Torchvision ResNet-18 backbone wrapped with skorch and 512-d feature extraction."
        ),
    ),
    ModelInfo(
        id="text_mlp",
        name="TextMLP (Dense Embedding Network)",
        architecture="Linear(64->128) + BatchNorm + Dropout + Linear(128->64) + Classifier",
        supported_datasets=["agnews"],
        embedding_dim=64,
        supports_mc_dropout=True,
        description=(
            "Dense multi-layer perceptron tailored for 64-d AG News embeddings with 64-d bottleneck features."
        ),
    ),
]


_STRATEGY_DETAILS = {
    "random": {
        "name": "Uniform Random Sampling",
        "category": "Baseline",
        "formula": "x ~ Uniform(U)",
        "description": "Selects K unlabeled samples uniformly at random as the passive learning baseline.",
        "requires_embeddings": False,
        "requires_mc_dropout": False,
    },
    "least_confidence": {
        "name": "Least Confidence",
        "category": "Uncertainty",
        "formula": "score(x) = 1 - max_c P(y=c | x)",
        "description": "Queries samples whose top predicted class has the lowest calibrated probability.",
        "requires_embeddings": False,
        "requires_mc_dropout": False,
    },
    "margin": {
        "name": "Margin Sampling",
        "category": "Uncertainty",
        "formula": "score(x) = P(y_(1) | x) - P(y_(2) | x)",
        "description": "Selects samples with the smallest margin between the top-1 and top-2 predicted classes.",
        "requires_embeddings": False,
        "requires_mc_dropout": False,
    },
    "entropy": {
        "name": "Shannon Predictive Entropy",
        "category": "Uncertainty",
        "formula": "H(x) = -sum_c P(y=c | x) log2 P(y=c | x)",
        "description": "Maximizes total predictive Shannon entropy across all class probabilities.",
        "requires_embeddings": False,
        "requires_mc_dropout": False,
    },
    "bald": {
        "name": "BALD (MC-Dropout Mutual Info)",
        "category": "Bayesian Uncertainty",
        "formula": "I[y, w | x] = H[E_t(p_t)] - E_t[H(p_t)]",
        "description": "Uses Monte Carlo Dropout forward passes to measure epistemic model disagreement.",
        "requires_embeddings": False,
        "requires_mc_dropout": True,
    },
    "coreset": {
        "name": "Core-Set (k-Center Greedy)",
        "category": "Diversity",
        "formula": "argmax_{i in U} min_{j in L U S} ||z_i - z_j||_2",
        "description": "Maximizes geometric coverage over the model's latent bottleneck embedding space.",
        "requires_embeddings": True,
        "requires_mc_dropout": False,
    },
    "badge": {
        "name": "BADGE (Gradient Embeddings)",
        "category": "Uncertainty + Diversity",
        "formula": "g_x = (p - 1_{y_hat}) * z(x) -> K-Means++",
        "description": "Samples diverse high-magnitude hypothetical gradient embeddings via K-Means++.",
        "requires_embeddings": True,
        "requires_mc_dropout": False,
    },
    "hybrid": {
        "name": "Hybrid (Top-3K Entropy + K-Means)",
        "category": "Uncertainty + Diversity",
        "formula": "Top-3K Entropy -> K-Means Medoids in Z",
        "description": "Filters the top-3K most uncertain candidates then picks K diverse cluster medoids.",
        "requires_embeddings": True,
        "requires_mc_dropout": False,
    },
    "cold_start_kmeans": {
        "name": "Cold-Start K-Means Medoids",
        "category": "Initialization",
        "formula": "K-Means Medoids on Raw Features",
        "description": "Selects representative cluster medoids before any labeled training data exists.",
        "requires_embeddings": False,
        "requires_mc_dropout": False,
    },
}


def get_datasets_catalog() -> List[DatasetSummary]:
    from backend.api.services.upload_service import list_custom_dataset_summaries
    custom_datasets = list_custom_dataset_summaries()
    return _DATASET_CATALOG + custom_datasets



def get_models_catalog() -> List[ModelInfo]:
    return _MODEL_CATALOG


def get_strategies_catalog() -> List[StrategyInfo]:
    registered = list_registered_strategies()
    items: List[StrategyInfo] = []
    for strat_id in registered:
        meta = _STRATEGY_DETAILS.get(
            strat_id,
            {
                "name": strat_id.replace("_", " ").title(),
                "category": "Registered",
                "formula": "Custom Strategy",
                "description": f"Registered active learning strategy '{strat_id}'.",
                "requires_embeddings": False,
                "requires_mc_dropout": False,
            },
        )
        items.append(StrategyInfo(id=strat_id, **meta))
    return items

