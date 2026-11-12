"""PyTorch neural network architectures for text embeddings and classification."""

import torch
import torch.nn as nn


class TextMLP(nn.Module):
    """Multi-layer perceptron for dense text embeddings."""

    def __init__(self, in_features: int = 64, num_classes: int = 4, dropout_rate: float = 0.3):
        super().__init__()
        self.embedding_dim = 64
        self.backbone = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(128, self.embedding_dim),
            nn.BatchNorm1d(self.embedding_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
        )
        self.classifier = nn.Linear(self.embedding_dim, num_classes)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.forward_features(x)
        return self.classifier(feat)
