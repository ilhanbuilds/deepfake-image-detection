"""Sabit V2 modelinde, aynı test fotoğraflarına eşleştirilmiş kalite deneyleri."""

import hashlib
import io
import json
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path
from time import perf_counter
from uuid import uuid4

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
from torchvision.transforms import functional as TF

from src.dataset import IMAGE_SIZE, make_loader, read_image
from src.evaluate import calculate_metrics
from src.experiment_log import append_experiment
from src.model import load_checkpoint, resolve_device
from src.presplit_dataset import resolve_data_root

CONDITIONS = (
    "CLEAN",
    "JPEG_Q70",
    "JPEG_Q50",
    "BLUR_SIGMA1",
    "BLUR_SIGMA2",
    "NOISE_STD0.01",
    "NOISE_STD0.03",
    "RESIZE_112",
)
NORMALIZATION = {"mean": (0.485, 0.456, 0.406), "std": (0.229, 0.224, 0.225)}


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


class RobustnessDataset(Dataset):
    """Sadece test dizini; label sırası filename sırasından bağımsızdır."""

    def __init__(self, dataset_root):
        self.root = resolve_data_root(dataset_root) / "test"
        folder = ImageFolder(str(self.root))
        if set(folder.class_to_idx) != {"real", "fake"}:
            raise ValueError("Test yalnızca real/fake sınıfları içermeli.")
        self.samples = []
        for filename, _ in folder.samples:
            path = Path(filename)
            if not path.resolve().is_relative_to(self.root.resolve()):
                raise ValueError("Test dosyası dataset dışına çıkıyor.")
            relative = path.relative_to(self.root).as_posix()
            self.samples.append(
                (relative, {"real": 0, "fake": 1}[Path(relative).parts[0]])
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        relative, label = self.samples[index]
        image = TF.resize(
            read_image(self.root / relative), [IMAGE_SIZE, IMAGE_SIZE], antialias=True
        )
        return TF.to_tensor(image), label, relative

    def fingerprint(self):
        value = hashlib.sha256()
        for relative, label in self.samples:
            value.update(f"{relative}:{label}:".encode())
            value.update(bytes.fromhex(digest(self.root / relative)))
        return value.hexdigest()


def corrupt_batch(clean, condition, paths, seed=42):
    """RGB [0,1], normalization öncesi. Girdi değiştirilmez.

    Noise seed relative path'e bağlıdır, batch/workers sırası etkilemez.
    İki noise koşulu aynı standart-normal örneği farklı std ile ölçekler.
    JPEG RGB/224, quality 70/50 ve 4:2:0 subsampling kullanır.
    Gaussian kernel 3 sigma yarıçaplı, reflect padding; resize bilinear/antialias.
    """
    if condition not in CONDITIONS or clean.ndim != 4 or len(clean) != len(paths):
        raise ValueError("Koşul, batch veya path sayısı geçersiz.")
    if condition == "CLEAN":
        return clean.clone()
    if condition.startswith("JPEG_"):
        quality = 70 if condition == "JPEG_Q70" else 50
        images = []
        for tensor in clean:
            with io.BytesIO() as buffer:
                TF.to_pil_image(tensor.cpu()).save(
                    buffer, format="JPEG", quality=quality, subsampling=2
                )
                buffer.seek(0)
                with Image.open(buffer) as image:
                    images.append(TF.to_tensor(image.convert("RGB")))
        return torch.stack(images).to(clean.device)
    if condition.startswith("BLUR_"):
        sigma = 1.0 if condition == "BLUR_SIGMA1" else 2.0
        kernel = 2 * math.ceil(3 * sigma) + 1
        return TF.gaussian_blur(clean, [kernel, kernel], [sigma, sigma])
    if condition.startswith("NOISE_"):
        std = 0.01 if condition == "NOISE_STD0.01" else 0.03
        noise = []
        for relative in paths:
            value = hashlib.sha256(
                f"{seed}:{relative}:gaussian_noise".encode()
            ).digest()
            generator = torch.Generator().manual_seed(
                int.from_bytes(value[:8], "big") & ((1 << 63) - 1)
            )
            noise.append(
                torch.randn(clean.shape[1:], generator=generator, dtype=clean.dtype)
            )
        return (clean + torch.stack(noise).to(clean.device) * std).clamp(0, 1)
    downsample = TF.resize(clean, [112, 112], antialias=True)
    return TF.resize(downsample, [IMAGE_SIZE, IMAGE_SIZE], antialias=True)


def compare_to_clean(results):
    clean = results["CLEAN"]
    for condition, metrics in results.items():
        for metric in ("accuracy", "f1", "roc_auc"):
            drop = clean[metric] - metrics[metric]
            metrics[f"{metric}_drop"] = drop
            if metric != "roc_auc":
                metrics[f"{metric}_drop_percentage_points"] = drop * 100
    return results


def save_report(report, output):
    output = Path(output)
    metrics, figures = output / "metrics", output / "figures"
    metrics.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    (metrics / "robustness_v2.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    rows = []
    for condition, result in report["results"].items():
        row = {"condition": condition, **result}
        row["confusion_matrix"] = json.dumps(row["confusion_matrix"])
        row["class_names"] = json.dumps(row["class_names"])
        rows.append(row)
    pd.DataFrame(rows).to_csv(metrics / "robustness_v2.csv", index=False)
    fig, ax = plt.subplots(figsize=(12, 5))
    for metric, label in (
        ("accuracy", "Accuracy"),
        ("f1", "F1 (FAKE)"),
        ("roc_auc", "ROC-AUC"),
    ):
        ax.plot(
            range(len(CONDITIONS)),
            [report["results"][c][metric] for c in CONDITIONS],
            marker="o",
            label=label,
        )
    ax.set_xticks(range(len(CONDITIONS)), CONDITIONS, rotation=25, ha="right")
    ax.set(
        ylim=(0, 1.02),
        ylabel="Skor",
        title="V2: aynı test fotoğraflarında kalite dayanıklılığı",
    )
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures / "robustness_v2.png", dpi=150)
    plt.close(fig)


def run_robustness(
    dataset_root,
    checkpoint_path,
    reference_path,
    output,
    log_path,
    device="cuda",
    batch_size=8,
    seed=42,
    expected_count=20000,
):
    output, checkpoint_path, reference_path = (
        Path(output),
        Path(checkpoint_path),
        Path(reference_path),
    )
    destinations = [
        output / "metrics/robustness_v2.json",
        output / "metrics/robustness_v2.csv",
        output / "figures/robustness_v2.png",
    ]
    if any(path.exists() for path in destinations):
        raise FileExistsError(
            "Mevcut robustness çıktıları korunuyor; yeni --output-dir kullanın."
        )
    if batch_size < 1:
        raise ValueError("Batch size pozitif olmalı.")
    started = perf_counter()
    date = datetime.now(timezone(timedelta(hours=3))).isoformat(timespec="seconds")
    record = {
        "id": f"robustness-v2-{uuid4().hex}",
        "name": "V2 fotoğraf kalite dayanıklılığı",
        "date": date,
        "model": "EfficientNet-B0",
        "checkpoint": str(checkpoint_path),
        "dataset": str(dataset_root),
        "counts": {
            "train": "kullanılmadı",
            "val": "kullanılmadı",
            "test": expected_count,
        },
        "objective": "Sabit modelin aynı fotoğraflarda kalite bozulmasına dayanıklılığı",
        "settings": {
            "image_size": 224,
            "batch_size": batch_size,
            "seed": seed,
            "learning_rate": "uygulanmaz",
            "epoch": "uygulanmaz",
            "conditions": list(CONDITIONS),
            "preprocessing": "RGB/EXIF -> bilinear 224 -> corruption -> ImageNet normalization",
            "jpeg": {"qualities": [70, 50], "subsampling": "4:2:0"},
            "blur": {"sigmas": [1, 2], "kernels": [7, 13], "padding": "reflect"},
            "noise": {
                "stds": [0.01, 0.03],
                "domain": "RGB [0,1], clamp [0,1]",
                "seed": "SHA256(seed:relative_path:gaussian_noise), CPU generator",
            },
            "resize": "224 -> 112 -> 224, bilinear, antialias",
        },
        "results": {},
        "duration_seconds": 0,
        "observations": "",
        "conclusion": "",
        "files": [],
        "status": "failed",
    }
    try:
        checkpoint_sha = digest(checkpoint_path)
        reference_sha = digest(reference_path)
        reference = json.loads(reference_path.read_text(encoding="utf-8"))
        if (
            reference.get("checkpoint_sha256") != checkpoint_sha
            or reference.get("label_mapping") != {"REAL": 0, "FAKE": 1}
            or reference.get("test_image_count") != expected_count
        ):
            raise ValueError(
                "Checkpoint/etiket/test sayısı final test referansıyla uyuşmuyor."
            )
        dataset = RobustnessDataset(dataset_root)
        counts = {
            name: sum(label == index for _, label in dataset.samples)
            for index, name in enumerate(("REAL", "FAKE"))
        }
        if len(dataset) != expected_count or counts != reference["test_class_counts"]:
            raise ValueError("Test sınıf dağılımı veya dosya sayısı değişmiş.")
        fingerprint = dataset.fingerprint()
        if fingerprint != reference["test_fingerprint"]:
            raise ValueError("Test içeriği final test sonrasında değişmiş.")
        resolved_device = resolve_device(device)
        model, checkpoint = load_checkpoint(checkpoint_path, resolved_device)
        if (
            checkpoint.get("pipeline") != "presplit_image_v2"
            or checkpoint["epoch"] != reference["best_epoch"]
        ):
            raise ValueError("Checkpoint pipeline/best epoch uyuşmuyor.")
        loader = make_loader(
            dataset,
            batch_size,
            workers=0,
            shuffle=False,
            pin_memory=resolved_device.type == "cuda",
            seed=seed,
        )
        targets, predictions, scores = (
            [],
            {c: [] for c in CONDITIONS},
            {c: [] for c in CONDITIONS},
        )
        condition_durations = {c: 0.0 for c in CONDITIONS}
        with torch.inference_mode():
            model.eval()
            for batch, (images, labels, paths) in enumerate(loader, 1):
                targets.extend(labels.tolist())
                images = images.to(resolved_device)
                for condition in CONDITIONS:
                    if resolved_device.type == "cuda":
                        torch.cuda.synchronize(resolved_device)
                    begin = perf_counter()
                    degraded = corrupt_batch(images, condition, paths, seed)
                    logits = model(TF.normalize(degraded, **NORMALIZATION))
                    if not torch.isfinite(logits).all().item():
                        raise ValueError("Inference logit NaN/Inf.")
                    predictions[condition].extend(logits.argmax(dim=1).cpu().tolist())
                    scores[condition].extend(
                        torch.softmax(logits, dim=1)[:, 1].cpu().tolist()
                    )
                    condition_durations[condition] += perf_counter() - begin
                if batch % 100 == 0:
                    print(
                        f"Tüm 8 koşul: {len(targets)}/{len(dataset)} fotoğraf",
                        flush=True,
                    )
        results = compare_to_clean(
            {
                c: {
                    **calculate_metrics(targets, predictions[c], scores[c]),
                    "duration_seconds": condition_durations[c],
                }
                for c in CONDITIONS
            }
        )
        if (
            digest(checkpoint_path) != checkpoint_sha
            or digest(reference_path) != reference_sha
        ):
            raise RuntimeError(
                "Checkpoint veya final test referansı işlem sırasında değişti."
            )
        report = {
            "date": date,
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": checkpoint_sha,
            "best_epoch": checkpoint["epoch"],
            "dataset_root": str(dataset.root),
            "test_fingerprint": fingerprint,
            "test_image_count": len(dataset),
            "class_counts": counts,
            "label_mapping": {"REAL": 0, "FAKE": 1},
            "seed": seed,
            "device": str(resolved_device),
            "evaluation_precision": "float32",
            "results": results,
            "settings": record["settings"],
            "total_duration_seconds": perf_counter() - started,
            "reference_clean_confusion_matrix_matches": results["CLEAN"][
                "confusion_matrix"
            ]
            == reference["confusion_matrix"],
            "reference_clean_metric_differences": {
                key: results["CLEAN"][key] - reference[key]
                for key in ("accuracy", "f1", "roc_auc")
            },
            "test_used_for_tuning": False,
            "normalization": NORMALIZATION,
        }
        save_report(report, output)
        record.update(
            status="completed",
            results=results,
            observations="CLEAN ile eşleştirilmiş kalite koşulları; negatif düşüş iyileşme anlamına gelir.",
            conclusion="Sonuçlar betimleyicidir; threshold/model tuning yapılmadı.",
            files=[str(p) for p in destinations],
        )
        return report
    except Exception as error:
        record.update(
            observations=f"{type(error).__name__}: {error}",
            conclusion="Deney tamamlanamadı; metrik uydurulmadı.",
        )
        raise
    finally:
        record["duration_seconds"] = perf_counter() - started
        append_experiment(Path(log_path), record)
