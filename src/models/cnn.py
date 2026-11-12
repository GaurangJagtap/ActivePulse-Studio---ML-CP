"""PyTorch convolutional neural networks for image classification."""

import torch
import torch.nn as nn
from torchvision import models
from typing import Optional


class CIFARConvNet(nn.Module):
    """Convolutional Neural Network tailored for 32x32 images (CIFAR-10).
    
    Includes dropout layers for Monte-Carlo Dropout uncertainty estimation,
    and a forward_features method for embedding extraction.
    """

    def __init__(self, num_classes: int = 10, dropout_rate: float = 0.3):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 16x16
            nn.Dropout2d(p=dropout_rate / 2),

            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 8x8
            nn.Dropout2d(p=dropout_rate / 2),

            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((2, 2)),  # 128 * 2 * 2 = 512
        )

        self.embedding_dim = 128
        self.fc_embed = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 2 * 2, self.embedding_dim),
            nn.BatchNorm1d(self.embedding_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
        )

        self.classifier = nn.Linear(self.embedding_dim, num_classes)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract bottleneck embedding representation."""
        conv_out = self.features(x)
        return self.fc_embed(conv_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning unnormalized logits."""
        embed = self.forward_features(x)
        return self.classifier(embed)


class ResNet18TransferNet(nn.Module):
    """ResNet-18 backbone with pretrained weights and MC-Dropout classification head."""

    def __init__(self, num_classes: int = 10, pretrained: bool = False, dropout_rate: float = 0.3):
        super().__init__()
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        backbone = models.resnet18(weights=weights)

        # Replace first conv for 32x32 if needed or keep standard
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])
        in_features = backbone.fc.in_features
        self.embedding_dim = in_features

        self.dropout = nn.Dropout(p=dropout_rate)
        self.classifier = nn.Linear(in_features, num_classes)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(x)
        feat = torch.flatten(feat, 1)
        return feat

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.forward_features(x)
        feat = self.dropout(feat)
        return self.classifier(feat)
