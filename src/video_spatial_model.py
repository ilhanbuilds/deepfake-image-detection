"""Order-invariant image backbone + masked feature mean + one video logit."""

import torch
from torch import nn
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

from src.video_finetune_model import PartialVideoClassifier
from src.video_model import VideoClassifier

PIPELINE = "video_spatial_mean_v1"


def masked_feature_mean(features, lengths):
    if (
        features.ndim != 3
        or features.shape[2] != 1280
        or lengths.shape != (features.shape[0],)
        or lengths.dtype not in (torch.int32, torch.int64)
        or torch.any(lengths < 3)
        or torch.any(lengths > 6)
        or torch.any(lengths > features.shape[1])
    ):
        raise ValueError("Spatial mean: [B,T,1280], integer lengths 3–6 gerekli.")
    mask = (
        torch.arange(features.shape[1], device=features.device)[None, :]
        < lengths.to(features.device)[:, None]
    )
    # Multiplication by zero would preserve NaN padding. Select zero instead.
    valid = torch.where(mask[..., None], features.float(), 0.0)
    if not torch.isfinite(valid).all():
        raise ValueError("Geçerli spatial feature NaN/Inf içeriyor.")
    return valid.sum(dim=1) / lengths.to(features.device, torch.float32)[:, None]


class SpatialVideoClassifier(nn.Module):
    feature_dim = 1280

    def __init__(self, pretrained=True, backbone_mode="partial"):
        super().__init__()
        if backbone_mode not in ("frozen", "partial"):
            raise ValueError("Backbone mode frozen veya partial olmalı.")
        b0 = efficientnet_b0(
            weights=EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        )
        self.backbone = nn.Sequential(b0.features, b0.avgpool, nn.Flatten(1))
        self.backbone.requires_grad_(False)
        self.backbone_mode = backbone_mode
        if backbone_mode == "partial":
            self.backbone[0][8].requires_grad_(True)
        self.head = nn.Linear(1280, 1)
        self.train()

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()
        if self.backbone_mode == "partial":
            self.backbone[0][8].train(mode)
        for module in self.backbone.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.eval()
        return self

    def backbone_fingerprint(self):
        return VideoClassifier.backbone_fingerprint(self)

    def extract_features(self, frames, chunk_size=2):
        # Reuse the exact image-input/chunk/prefix pipeline, without a GRU instance.
        return PartialVideoClassifier.extract_features(self, frames, chunk_size)

    def forward(self, features, lengths):
        return self.head(masked_feature_mean(features, lengths)).squeeze(-1)


def make_spatial_optimizer(model):
    groups = [dict(params=model.head.parameters(), lr=3e-4)]
    if model.backbone_mode == "partial":
        groups.insert(0, dict(params=model.backbone[0][8].parameters(), lr=1e-5))
    return torch.optim.AdamW(groups, weight_decay=0.01)
