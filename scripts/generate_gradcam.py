"""best_model.pth ile tek görüntü açıklaması üretir; eğitim başlatmaz."""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="REAL/FAKE tahmini, confidence ve Grad-CAM görselleri üretir."
    )
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=PROJECT_ROOT / "outputs/checkpoints/best_model.pth",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=PROJECT_ROOT / "outputs/figures/gradcam"
    )
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()
    # Script doğrudan veya repository dışındaki çalışma dizininden başlatılabilir.
    sys.path.insert(0, str(PROJECT_ROOT))
    from src.explainability import generate_gradcam, save_gradcam
    from src.model import load_checkpoint, resolve_device

    device = resolve_device(args.device)
    model, checkpoint = load_checkpoint(args.checkpoint, device)
    result = generate_gradcam(model, args.image, device)
    outputs = save_gradcam(
        result, args.output_dir, args.image, args.checkpoint, checkpoint["epoch"]
    )
    print(
        f"Tahmin: {result.label}; confidence: %{result.confidence_percent:.2f}; katman: {result.target_layer}"
    )
    if outputs["heatmap_empty"]:
        print(
            "Uyarı: Bu görüntü/sınıf için pozitif uzamsal Grad-CAM sinyali yok; harita sıfır."
        )
    print(json.dumps(outputs, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
