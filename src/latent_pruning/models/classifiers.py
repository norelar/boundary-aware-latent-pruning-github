"""Classifier factory retaining the two-layer research head from the notebooks."""

from __future__ import annotations

import torch
import torch.nn as nn


MODEL_ALIASES: dict[str, str] = {
    "resnet50": "resnet50",
    "xception": "xception",
    "vgg16": "vgg16",
    "efficientnet_b0": "efficientnet_b0",
    "vit_base": "vit_base_patch16_224",
    "vit_tiny": "vit_tiny_patch16_224",
    "deit_tiny": "deit_tiny_patch16_224",
    "swin_base": "swin_base_patch4_window7_224",
    "mobilevit_s": "mobilevit_s",
    "mobilevit_v2_050": "mobilevitv2_050",
    "pvt_v2_b0": "pvt_v2_b0",
    "tinyvit_5m": "tiny_vit_5m_224",
    "tinyvit_11m": "tiny_vit_11m_224",
    "mobilenet_v2": "mobilenetv2_100",
    "mobilenet_v3_small": "mobilenetv3_small_100",
}


class TimmClassifier(nn.Module):
    def __init__(
        self,
        model_name: str,
        num_classes: int,
        pretrained: bool = True,
        hidden_dim: int = 256,
        dropout: float = 0.30,
        freeze_backbone: bool = False,
    ):
        super().__init__()
        try:
            import timm
        except ImportError as exc:
            raise ImportError("Install experiment dependencies with: pip install -e '.[experiment]'") from exc

        resolved_name = MODEL_ALIASES.get(model_name, model_name)
        try:
            self.backbone = timm.create_model(
                resolved_name,
                pretrained=pretrained,
                num_classes=0,
                global_pool="avg",
            )
        except TypeError:
            self.backbone = timm.create_model(resolved_name, pretrained=pretrained, num_classes=0)
        feature_count = getattr(self.backbone, "num_features", None)
        if not feature_count:
            raise ValueError(f"Cannot determine output feature size for timm model {resolved_name!r}")
        if freeze_backbone:
            for parameter in self.backbone.parameters():
                parameter.requires_grad = False
        self.model_name = resolved_name
        self.classifier = nn.Sequential(
            nn.Linear(int(feature_count), hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.backbone(inputs)
        if isinstance(features, (tuple, list)):
            features = features[0]
        if isinstance(features, dict):
            features = next(iter(features.values()))
        if features.ndim == 4:
            features = features.mean(dim=(2, 3))
        elif features.ndim == 3:
            features = features.mean(dim=1)
        elif features.ndim > 2:
            features = torch.flatten(features, start_dim=1)
        return self.classifier(features)


def create_classifier(
    model_name: str,
    num_classes: int,
    pretrained: bool = True,
    hidden_dim: int = 256,
    dropout: float = 0.30,
    freeze_backbone: bool = False,
) -> nn.Module:
    return TimmClassifier(
        model_name=model_name,
        num_classes=num_classes,
        pretrained=pretrained,
        hidden_dim=hidden_dim,
        dropout=dropout,
        freeze_backbone=freeze_backbone,
    )
