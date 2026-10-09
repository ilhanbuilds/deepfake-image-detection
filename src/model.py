"""EfficientNet-B0 ve checkpoint yükleme."""

import argparse
from pathlib import Path

import torch
from torch import nn
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

from src.dataset import CLASS_NAMES, IMAGE_SIZE


def build_model(pretrained: bool = True) -> nn.Module:
    weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
    model = efficientnet_b0(weights=weights)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(CLASS_NAMES))
    return model


def resolve_device(requested: str = "auto") -> torch.device:
    if requested not in {"auto", "cpu", "cuda"}:
        raise ValueError("Device auto, cpu veya cuda olmalı.")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA kullanılamıyor. CUDA destekli PyTorch kurulumunu kontrol edin."
        )
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(requested)


def load_checkpoint(path: str | Path, device: torch.device) -> tuple[nn.Module, dict]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if (
        checkpoint.get("class_names") != list(CLASS_NAMES)
        or checkpoint.get("image_size") != IMAGE_SIZE
    ):
        raise ValueError(
            "Checkpoint sınıf sırası veya görüntü boyutu bu projeyle uyumlu değil."
        )
    model = build_model(pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model.to(device).eval(), checkpoint


def environment_info() -> dict:
    import platform

    info = {
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
    }
    if info["cuda_available"]:
        props = torch.cuda.get_device_properties(0)
        info.update(gpu=props.name, vram_gib=round(props.total_memory / 1024**3, 2))
    return info


def main() -> None:
    import json

    parser = argparse.ArgumentParser(
        description="Eğitim yapmadan Python/CUDA ortamını kontrol eder."
    )
    parser.parse_args()
    print(json.dumps(environment_info(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
