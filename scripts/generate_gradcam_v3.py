"""Testten seed 42 ile ilk beş doğru REAL/FAKE açıklamasını seçer; tuning yapmaz."""

import argparse
import json
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def select_correct_examples(model, dataset, device, count=5, seed=42):
    """Sınıf başına rastgele sıradaki ilk doğru örnekler; confidence'a göre sıralamaz."""
    from src.dataset import CLASS_NAMES
    from src.predict import predict_image

    if count < 1:
        raise ValueError("Örnek sayısı pozitif olmalı.")
    rng = random.Random(seed)
    selected, examined = [], {}
    for label, name in enumerate(CLASS_NAMES):
        candidates = [item for item in dataset.samples if item[1] == label]
        rng.shuffle(candidates)
        examined[name] = 0
        accepted = 0
        for relative, _ in candidates:
            path = dataset.root / relative
            prediction = predict_image(model, path, device)
            examined[name] += 1
            if prediction["label"] == name:
                selected.append((path, label))
                accepted += 1
                if accepted == count:
                    break
        if accepted != count:
            raise ValueError(f"{name} sınıfında {count} doğru örnek bulunamadı.")
    return selected, examined


def main() -> None:
    sys.path.insert(0, str(PROJECT_ROOT))
    from src.dataset import CLASS_NAMES
    from src.explainability import generate_gradcam, save_gradcam
    from src.model import load_checkpoint, resolve_device
    from src.robustness import RobustnessDataset, digest

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root", type=Path, default=PROJECT_ROOT / "data/real_vs_fake"
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=PROJECT_ROOT / "outputs/checkpoints/best_model_v3_robust.pth",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs/figures/gradcam_v3",
    )
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()
    reference = json.loads(
        (PROJECT_ROOT / "outputs/metrics/final_test_metrics_v3.json").read_text("utf-8")
    )
    dataset = RobustnessDataset(args.data_root)
    checkpoint_hash = digest(args.checkpoint)
    if checkpoint_hash != reference["checkpoint_sha256"]:
        raise ValueError("Checkpoint final V3 değerlendirmesiyle eşleşmiyor.")
    fingerprint = dataset.fingerprint()
    if fingerprint != reference["test_fingerprint"]:
        raise ValueError("Test fotoğrafları final V3 değerlendirmesiyle eşleşmiyor.")
    device = resolve_device(args.device)
    model, checkpoint = load_checkpoint(args.checkpoint, device)
    selected, examined = select_correct_examples(model, dataset, device)
    outputs = []
    for path, label in selected:
        result = generate_gradcam(model, path, device)
        if result.label != CLASS_NAMES[label]:
            raise RuntimeError("Grad-CAM tahmini seçim tahminiyle eşleşmiyor.")
        outputs.append(
            {
                "true_label": CLASS_NAMES[label],
                **save_gradcam(
                    result, args.output_dir, path, args.checkpoint, checkpoint["epoch"]
                ),
            }
        )
        print(
            f"{path.name}: {result.label} %{result.confidence_percent:.2f}", flush=True
        )
    summary = {
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": checkpoint_hash,
        "best_epoch": checkpoint["epoch"],
        "test_fingerprint": fingerprint,
        "seed": 42,
        "device": str(device),
        "examined": examined,
        "selection": "Seed 42; her sınıfta ilk 5 doğru tahmin; confidence sıralaması yok.",
        "purpose": "Açıklayıcı doğru örnekler; genel performans tahmini veya tuning değildir.",
        "examples": outputs,
    }
    # Önceki çalıştırmanın özetini de korur.
    summary_path = Path(outputs[0]["metadata_file"]).parent / "selection_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), "utf-8")
    print(f"10 Grad-CAM tamamlandı. Özet: {summary_path}", flush=True)


if __name__ == "__main__":
    main()
