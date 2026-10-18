"""Separate image-input model: final EfficientNet feature block + packed GRU."""

import torch
from torch import nn

from src.video_model import VideoClassifier

PIPELINE = "video_partial_finetune_v1"


class PartialVideoClassifier(VideoClassifier):
    def __init__(self, pretrained: bool = True, hidden_size: int = 64):
        super().__init__(pretrained=pretrained, hidden_size=hidden_size)
        # Torchvision B0 features[8]: final 320 -> 1280 1x1 conv + BN + SiLU.
        # The preceding MBConv stage features[7] remains frozen.
        self.backbone[0][-1].requires_grad_(True)
        self.train()

    def train(self, mode: bool = True):
        super().train(mode)
        self.backbone[0][-1].train(mode)
        # Keep ALL running stats fixed, including the last block's BN.
        # Its affine weight/bias can still receive gradients in eval mode.
        for module in self.backbone.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.eval()
        return self

    def extract_features(self, frames: torch.Tensor, chunk_size: int = 2):
        if (
            frames.ndim != 4
            or frames.shape[1] != 3
            or not 3 <= len(frames) <= 6
            or frames.shape[2:] != (160, 160)
            or frames.dtype != torch.float32
            or not torch.isfinite(frames).all()
            or chunk_size < 1
        ):
            raise ValueError("Fine-tuning gerçek FP32 [3..6,3,160,160] görüntü ister.")
        sequences = []
        for chunk in frames.split(chunk_size):
            # No activation graph is needed through the frozen prefix.
            with torch.no_grad():
                prefix = self.backbone[0][:-1](chunk)
            # Deliberately OUTSIDE no_grad: actual image activations train block 8.
            output = self.backbone[0][-1](prefix)
            sequences.append(self.backbone[2](self.backbone[1](output)))
        return torch.cat(sequences).float()


def make_optimizer(model, backbone_lr=1e-5, head_lr=3e-4, weight_decay=0.01):
    return torch.optim.AdamW(
        [
            dict(params=model.backbone[0][-1].parameters(), lr=backbone_lr),
            dict(
                params=[*model.gru.parameters(), *model.head.parameters()], lr=head_lr
            ),
        ],
        weight_decay=weight_decay,
    )
