"""Eğitilmiş checkpoint ile tek yüz görüntüsü tahmini."""

import argparse
import json
from pathlib import Path

import torch

from src.dataset import CLASS_NAMES, image_transform, read_image
from src.model import load_checkpoint, resolve_device


@torch.inference_mode()
def predict_image(
    model: torch.nn.Module, image_path: str | Path, device: torch.device
) -> dict:
    model.eval()
    tensor = image_transform()(read_image(image_path)).unsqueeze(0).to(device)
    probabilities = torch.softmax(model(tensor), dim=1)[0].cpu().tolist()
    index = max(range(len(CLASS_NAMES)), key=probabilities.__getitem__)
    return {
        "label": CLASS_NAMES[index],
        "probabilities": dict(zip(CLASS_NAMES, probabilities, strict=True)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Tek görüntü için REAL/FAKE tahmini.")
    parser.add_argument("image", type=Path)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()
    device = resolve_device(args.device)
    model, _ = load_checkpoint(args.checkpoint, device)
    print(json.dumps(predict_image(model, args.image, device), indent=2))


if __name__ == "__main__":
    main()
