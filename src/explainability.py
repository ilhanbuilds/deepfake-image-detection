"""Sabit modelin tahmin ettiği sınıf için Grad-CAM; ağırlıkları güncellemez."""

import json
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import colormaps
from PIL import Image
from torch import nn
from torch.nn import functional as F
from torchvision.models.efficientnet import EfficientNet

from src.dataset import CLASS_NAMES, image_transform, read_image


@dataclass
class GradCAMResult:
    label: str
    confidence_percent: float
    probabilities: dict[str, float]
    target_layer: str
    original: Image.Image
    heatmap: np.ndarray


def select_target_layer(model: nn.Module) -> tuple[str, nn.Conv2d]:
    """EfficientNet özellik çıkarıcısının son Conv2d katmanını yapısından seçer."""
    if not isinstance(model, EfficientNet):
        raise ValueError(
            "Otomatik katman seçimi torchvision EfficientNet modeli gerektirir."
        )
    candidates = [
        (f"features.{name}", layer)
        for name, layer in model.features.named_modules()
        if isinstance(layer, nn.Conv2d)
    ]
    if not candidates:
        raise ValueError("EfficientNet features içinde convolution katmanı bulunamadı.")
    name, layer = candidates[-1]
    linear_layers = [
        module for module in model.classifier.modules() if isinstance(module, nn.Linear)
    ]
    if not linear_layers or layer.out_channels != linear_layers[0].in_features:
        raise ValueError(
            "Son convolution sınıflandırıcı girdisiyle uyumlu değil; güvenli katman seçilemedi."
        )
    return name, layer


def generate_gradcam(
    model: nn.Module, image_path: str | Path, device: torch.device
) -> GradCAMResult:
    """Logit gradyanlarını kullanır; parametre .grad alanlarını veya model durumunu değiştirmez."""
    original = read_image(image_path)
    layer_name, layer = select_target_layer(model)
    modes = {module: module.training for module in model.modules()}
    activation: torch.Tensor | None = None

    def capture_activation(
        module: nn.Module, inputs: tuple, output: torch.Tensor
    ) -> None:
        nonlocal activation
        if (
            not isinstance(output, torch.Tensor)
            or output.ndim != 4
            or min(output.shape[-2:]) <= 1
        ):
            raise ValueError(
                "Seçilen convolution uzamsal NCHW özellik haritası üretmiyor."
            )
        activation = output

    handle = layer.register_forward_hook(capture_activation)
    try:
        model.eval()
        # Grad-CAM için gradyan gerekir; inference_mode/no_grad altında da çalışır.
        with (
            torch.inference_mode(False),
            torch.enable_grad(),
            torch.autocast(device_type=device.type, enabled=False),
        ):
            tensor = (
                image_transform()(original).unsqueeze(0).to(device).requires_grad_(True)
            )
            logits = model(tensor)
            if (
                logits.shape != (1, len(CLASS_NAMES))
                or not torch.isfinite(logits).all().item()
            ):
                raise ValueError("Model sonlu REAL/FAKE logitleri üretmeli.")
            if activation is None or not activation.requires_grad:
                raise RuntimeError(
                    "Convolution aktivasyonu veya gradyan bağlantısı yakalanamadı."
                )
            class_index = int(logits.argmax(dim=1).item())
            probabilities = torch.softmax(logits.detach(), dim=1)[0].cpu().tolist()
            gradients = torch.autograd.grad(logits[0, class_index], activation)[0]
            if (
                not torch.isfinite(activation).all().item()
                or not torch.isfinite(gradients).all().item()
            ):
                raise RuntimeError(
                    "Grad-CAM aktivasyonları/gradyanları NaN veya Inf içeriyor."
                )
            weights = gradients.mean(dim=(2, 3), keepdim=True)
            cam = torch.relu((weights * activation.detach()).sum(dim=1, keepdim=True))
            cam = F.interpolate(
                cam,
                size=(original.height, original.width),
                mode="bilinear",
                align_corners=False,
            )[0, 0]
            if not torch.isfinite(cam).all().item():
                raise RuntimeError("Grad-CAM haritası NaN veya Inf içeriyor.")
            cam = cam - cam.min()
            maximum = cam.max()
            cam = cam / maximum if maximum.item() > 0 else torch.zeros_like(cam)
            heatmap = cam.detach().float().cpu().numpy()
        return GradCAMResult(
            label=CLASS_NAMES[class_index],
            confidence_percent=probabilities[class_index] * 100,
            probabilities=dict(zip(CLASS_NAMES, probabilities, strict=True)),
            target_layer=layer_name,
            original=original,
            heatmap=heatmap,
        )
    finally:
        handle.remove()
        activation = None
        for module, mode in modes.items():
            module.training = mode


def save_gradcam(
    result: GradCAMResult,
    output_dir: str | Path,
    image_path: str | Path,
    checkpoint_path: str | Path,
    best_epoch: int,
) -> dict:
    """Her çalıştırmaya ayrı dizin açarak önceki açıklamaları korur."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = re.sub(r"[^A-Za-z0-9_-]", "_", Path(image_path).stem)[:60] or "image"
    run_dir = Path(tempfile.mkdtemp(prefix=f"gradcam_{stem}_", dir=output_dir))
    colored = np.round(colormaps["jet"](result.heatmap)[..., :3] * 255).astype(np.uint8)
    heatmap_image = Image.fromarray(colored)
    overlay = Image.blend(result.original, heatmap_image, alpha=0.4)
    paths = {
        name: run_dir / f"{name}.png"
        for name in ("original", "heatmap", "overlay", "comparison")
    }
    result.original.save(paths["original"])
    heatmap_image.save(paths["heatmap"])
    overlay.save(paths["overlay"])
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    try:
        for axis, picture, title in zip(
            axes,
            (result.original, heatmap_image, overlay),
            ("Orijinal", "Grad-CAM (kırmızı: yüksek)", "Bindirme"),
            strict=True,
        ):
            axis.imshow(picture)
            axis.set_title(title)
            axis.axis("off")
        fig.suptitle(f"{result.label} | Confidence: %{result.confidence_percent:.2f}")
        fig.tight_layout()
        fig.savefig(paths["comparison"], dpi=150)
    finally:
        plt.close(fig)
    metadata = {
        "label": result.label,
        "confidence_percent": result.confidence_percent,
        "probabilities": result.probabilities,
        "explained_class": result.label,
        "target_layer": result.target_layer,
        "best_epoch": best_epoch,
        "image": str(Path(image_path).resolve()),
        "checkpoint": str(Path(checkpoint_path).resolve()),
        "heatmap_empty": bool(not np.any(result.heatmap)),
        "files": {name: str(path.resolve()) for name, path in paths.items()},
    }
    metadata_path = run_dir / "prediction.json"
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return {**metadata, "metadata_file": str(metadata_path.resolve())}
