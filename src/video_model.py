"""Frozen ImageNet EfficientNet-B0 + ordered GRU; exactly one logit per video."""

import hashlib

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0


class VideoClassifier(nn.Module):
    feature_dim = 1280

    def __init__(self, pretrained: bool = True, hidden_size: int = 128):
        super().__init__()
        if hidden_size < 1:
            raise ValueError("GRU hidden size pozitif olmalı.")
        backbone = efficientnet_b0(
            weights=EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        )
        self.backbone = nn.Sequential(
            backbone.features, backbone.avgpool, nn.Flatten(1)
        )
        self.backbone.requires_grad_(False).eval()
        self.gru = nn.GRU(self.feature_dim, hidden_size, batch_first=True)
        self.head = nn.Linear(hidden_size, 1)
        self.hidden_size = hidden_size

    def train(self, mode: bool = True):
        super().train(mode)
        # Frozen BN statistics and stochastic depth must stay frozen as well.
        self.backbone.eval()
        return self

    def backbone_fingerprint(self) -> str:
        digest = hashlib.sha256()
        for name, tensor in self.backbone.state_dict().items():
            digest.update(name.encode())
            digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
        return digest.hexdigest()

    def extract_features(
        self, frames: torch.Tensor, chunk_size: int = 2
    ) -> torch.Tensor:
        if frames.ndim != 4 or frames.shape[1] != 3 or chunk_size < 1:
            raise ValueError("Frame tensor [T,3,H,W] ve chunk size pozitif olmalı.")
        with torch.no_grad():
            return torch.cat(
                [self.backbone(chunk) for chunk in frames.split(chunk_size)]
            ).float()

    def forward(self, features: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        if features.ndim != 3 or features.shape[-1] != self.feature_dim:
            raise ValueError("GRU girdisi [B,T,1280] olmalı.")
        if (
            lengths.shape != (features.shape[0],)
            or torch.any(lengths < 1)
            or torch.any(lengths > features.shape[1])
        ):
            raise ValueError("Video sequence uzunlukları geçersiz.")
        packed = pack_padded_sequence(
            features.float(), lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, hidden = self.gru(packed)
        return self.head(hidden[-1]).squeeze(-1)
