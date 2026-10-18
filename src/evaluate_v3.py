"""Sabit V3 checkpoint: önce CLEAN final test, sonra aynı V2 kalite protokolü."""

import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from time import perf_counter
from uuid import uuid4

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch
from torchvision.transforms import functional as TF

from src.dataset import make_loader
from src.evaluate import calculate_metrics, save_metrics
from src.experiment_log import append_experiment
from src.model import load_checkpoint, resolve_device
from src.robustness import (
    CONDITIONS,
    NORMALIZATION,
    RobustnessDataset,
    compare_to_clean,
    corrupt_batch,
    digest,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@torch.inference_mode()
def evaluate_conditions(model, loader, device, conditions, seed, stage):
    model.eval()
    targets = []
    predictions = {condition: [] for condition in conditions}
    scores = {condition: [] for condition in conditions}
    durations = {condition: 0.0 for condition in conditions}
    for batch, (images, labels, paths) in enumerate(loader, 1):
        targets.extend(labels.tolist())
        images = images.to(device)
        for condition in conditions:
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            started = perf_counter()
            transformed = corrupt_batch(images, condition, paths, seed)
            logits = model(TF.normalize(transformed, **NORMALIZATION))
            if not torch.isfinite(logits).all().item():
                raise ValueError("Inference logit NaN/Inf.")
            predictions[condition].extend(logits.argmax(dim=1).cpu().tolist())
            scores[condition].extend(torch.softmax(logits, dim=1)[:, 1].cpu().tolist())
            durations[condition] += perf_counter() - started
        if batch % 100 == 0 or batch == len(loader):
            print(f"{stage}: {len(targets)}/{len(loader.dataset)} fotoğraf", flush=True)
    return {
        condition: {
            **calculate_metrics(targets, predictions[condition], scores[condition]),
            "duration_seconds": durations[condition],
        }
        for condition in conditions
    }


def compare_models(v2_results, v3_results):
    return {
        condition: {
            **{
                f"v{version}_{metric}": results[condition][metric]
                for version, results in ((2, v2_results), (3, v3_results))
                for metric in ("accuracy", "f1")
            },
            **{
                f"{metric}_improvement_percentage_points": (
                    v3_results[condition][metric] - v2_results[condition][metric]
                )
                * 100
                for metric in ("accuracy", "f1")
            },
        }
        for condition in CONDITIONS
    }


def save_comparison(report, output):
    output = Path(output)
    (output / "metrics").mkdir(parents=True, exist_ok=True)
    (output / "figures").mkdir(parents=True, exist_ok=True)
    (output / "metrics/robustness_v3.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    rows = []
    for condition, metrics in report["results"].items():
        row = {"condition": condition, **metrics, **report["v2_vs_v3"][condition]}
        for key in ("class_names", "confusion_matrix"):
            row[key] = json.dumps(row[key])
        rows.append(row)
    pd.DataFrame(rows).to_csv(output / "metrics/robustness_v3.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    for ax, metric, title in zip(axes, ("accuracy", "f1"), ("Accuracy", "F1 (FAKE)")):
        for version in (2, 3):
            ax.plot(
                range(len(CONDITIONS)),
                [
                    report["v2_vs_v3"][condition][f"v{version}_{metric}"] * 100
                    for condition in CONDITIONS
                ],
                marker="o",
                label=f"V{version}",
            )
        ax.set_xticks(range(len(CONDITIONS)), CONDITIONS, rotation=35, ha="right")
        ax.set(title=title, ylabel="Skor (%)", ylim=(0, 102))
        ax.grid(axis="y", alpha=0.3)
        ax.legend()
    fig.suptitle("V2 ve V3: aynı fotoğraflar, aynı kalite koşulları")
    fig.tight_layout()
    fig.savefig(output / "figures/robustness_v2_vs_v3.png", dpi=150)
    plt.close(fig)


def run_evaluation(
    dataset_root,
    checkpoint_path,
    v2_report_path,
    reference_path,
    output,
    log_path,
    device="cuda",
    batch_size=8,
    expected_count=20000,
):
    checkpoint_path, v2_report_path, reference_path, output = map(
        Path, (checkpoint_path, v2_report_path, reference_path, output)
    )
    destinations = [
        output / "metrics/final_test_metrics_v3.json",
        output / "metrics/robustness_v3.json",
        output / "metrics/robustness_v3.csv",
        output / "figures/robustness_v2_vs_v3.png",
        output / "figures/final_test_metrics_v3_confusion_matrix.png",
    ]
    if any(path.exists() for path in destinations):
        raise FileExistsError("Mevcut V3 değerlendirme çıktıları korunuyor.")
    if batch_size < 1:
        raise ValueError("Batch size pozitif olmalı.")
    started = perf_counter()
    date = datetime.now(timezone(timedelta(hours=3))).isoformat(timespec="seconds")
    record = {
        "id": f"v3-final-robustness-{uuid4().hex}",
        "name": "V3 CLEAN final test ve V2 ile eşleştirilmiş robustness",
        "date": date,
        "model": "EfficientNet-B0 V3 Robust",
        "checkpoint": str(checkpoint_path),
        "dataset": str(dataset_root),
        "counts": {
            "train": "okunmadı",
            "val": "okunmadı",
            "test_expected": expected_count,
        },
        "objective": "Sabit V3 modelini CLEAN test ve aynı 8 koşulda V2 ile karşılaştırmak",
        "settings": {},
        "results": {},
        "duration_seconds": 0,
        "observations": "",
        "conclusion": "",
        "files": [],
        "status": "failed",
    }
    try:
        protected = {
            path: digest(path)
            for path in (checkpoint_path, v2_report_path, reference_path)
        }
        v2 = json.loads(v2_report_path.read_text(encoding="utf-8"))
        reference = json.loads(reference_path.read_text(encoding="utf-8"))
        seed = v2["seed"]
        if (
            tuple(v2["results"]) != CONDITIONS
            or v2["test_fingerprint"] != reference["test_fingerprint"]
            or v2["checkpoint_sha256"] != reference["checkpoint_sha256"]
            or v2["label_mapping"] != {"REAL": 0, "FAKE": 1}
            or v2["evaluation_precision"] != "float32"
            or v2["settings"]["batch_size"] != batch_size
            or v2["normalization"] != {k: list(v) for k, v in NORMALIZATION.items()}
        ):
            raise ValueError("V2 referansı/protokolü uyumsuz.")
        dataset = RobustnessDataset(dataset_root)
        counts = {
            name: sum(label == index for _, label in dataset.samples)
            for index, name in enumerate(("REAL", "FAKE"))
        }
        fingerprint = dataset.fingerprint()
        if (
            len(dataset) != expected_count
            or len(dataset) != v2["test_image_count"]
            or counts != reference["test_class_counts"]
            or counts != v2["class_counts"]
            or fingerprint != reference["test_fingerprint"]
        ):
            raise ValueError("Aynı test fotoğrafları/sınıf dağılımı doğrulanamadı.")
        resolved_device = resolve_device(device)
        model, checkpoint = load_checkpoint(checkpoint_path, resolved_device)
        if (
            checkpoint.get("pipeline") != "presplit_image_v3_robust"
            or checkpoint["epoch"] != checkpoint["best_epoch"]
            or checkpoint["dataset_fingerprint"]
            != reference["training_validation_fingerprint"]
        ):
            raise ValueError(
                "V3 best checkpoint veya training dataset metadata'sı uyumsuz."
            )
        loader = make_loader(
            dataset,
            batch_size,
            workers=0,
            shuffle=False,
            pin_memory=resolved_device.type == "cuda",
            seed=seed,
        )
        record["counts"] = {
            "train": "okunmadı",
            "val": "okunmadı",
            "test": len(dataset),
            "test_classes": counts,
        }
        record["settings"] = {
            **v2["settings"],
            "best_epoch": checkpoint["epoch"],
            "device": str(resolved_device),
            "evaluation_precision": "float32",
        }
        metadata = {
            "date": date,
            "best_epoch": checkpoint["epoch"],
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": protected[checkpoint_path],
            "dataset_root": str(dataset.root),
            "test_fingerprint": fingerprint,
            "test_image_count": len(dataset),
            "test_class_counts": counts,
            "label_mapping": {"REAL": 0, "FAKE": 1},
            "device": str(resolved_device),
            "image_size": 224,
            "evaluation_precision": "float32",
            "seed": seed,
            "training_validation_fingerprint": checkpoint["dataset_fingerprint"],
            "test_used_for_training_or_checkpoint_selection": False,
            "test_used_for_tuning_in_this_evaluation": False,
            "test_previously_inspected_for_v3_augmentation_design": True,
        }
        print(
            f"V3 best epoch={checkpoint['epoch']}; mapping REAL=0 / FAKE=1; test={len(dataset)}",
            flush=True,
        )
        clean_started = perf_counter()
        clean = evaluate_conditions(
            model, loader, resolved_device, ("CLEAN",), seed, "CLEAN final test"
        )["CLEAN"]
        matrix = clean["confusion_matrix"]
        final = {
            **clean,
            **metadata,
            "split": "test",
            "correct_predictions": matrix[0][0] + matrix[1][1],
            "incorrect_predictions": matrix[0][1] + matrix[1][0],
            "inference_duration_seconds": perf_counter() - clean_started,
        }
        save_metrics(final, output, "final_test_metrics_v3")
        print("CLEAN final test kaydedildi: " + json.dumps(clean), flush=True)
        robustness_started = perf_counter()
        results = compare_to_clean(
            evaluate_conditions(
                model,
                loader,
                resolved_device,
                CONDITIONS,
                seed,
                "V3 robustness (8 koşul)",
            )
        )
        if any(
            results["CLEAN"][key] != clean[key]
            for key in (
                "accuracy",
                "precision",
                "recall",
                "f1",
                "roc_auc",
                "confusion_matrix",
            )
        ):
            raise RuntimeError("Robustness CLEAN ile ayrı final test eşleşmedi.")
        if any(digest(path) != sha for path, sha in protected.items()):
            raise RuntimeError("Checkpoint veya V2 referans dosyası değişti.")
        comparison = compare_models(v2["results"], results)
        report = {
            **metadata,
            "class_counts": counts,
            "results": results,
            "settings": record["settings"],
            "normalization": NORMALIZATION,
            "v2_report_sha256": protected[v2_report_path],
            "v2_checkpoint_sha256": v2["checkpoint_sha256"],
            "v2_vs_v3": comparison,
            "clean_matches_final_test": True,
            "robustness_duration_seconds": perf_counter() - robustness_started,
            "total_duration_seconds": perf_counter() - started,
        }
        save_comparison(report, output)
        shifts = {
            c: {
                "v2_fake_to_real": v2["results"][c]["confusion_matrix"][1][0],
                "v3_fake_to_real": results[c]["confusion_matrix"][1][0],
            }
            for c in ("BLUR_SIGMA1", "BLUR_SIGMA2", "RESIZE_112")
        }
        best_accuracy = max(
            comparison,
            key=lambda c: comparison[c]["accuracy_improvement_percentage_points"],
        )
        best_f1 = max(
            comparison, key=lambda c: comparison[c]["f1_improvement_percentage_points"]
        )
        record.update(
            status="completed",
            results={
                "clean_final_test": final,
                "robustness": results,
                "v2_vs_v3": comparison,
            },
            observations=json.dumps(
                {
                    "blur_resize_fake_to_real_counts": shifts,
                    "largest_accuracy_improvement": best_accuracy,
                    "largest_f1_improvement": best_f1,
                },
                ensure_ascii=False,
            ),
            conclusion="Aynı testte eşleştirilmiş karşılaştırma. Ağırlık/threshold değiştirilmedi. Test V3 augmentation tasarımı öncesinde görülmüştü; tamamen görülmemiş bağımsız test değildir.",
            files=[str(path) for path in destinations],
        )
        print(
            json.dumps({"clean": final, "comparison": comparison}, indent=2), flush=True
        )
        return report
    except Exception as error:
        record.update(
            observations=f"{type(error).__name__}: {error}",
            conclusion="Değerlendirme başarısız; metrik uydurulmadı.",
        )
        raise
    finally:
        record["duration_seconds"] = perf_counter() - started
        append_experiment(Path(log_path), record)


def main():
    parser = argparse.ArgumentParser(
        description="V3 final test ve V2 ile aynı robustness koşulları; eğitim yapmaz."
    )
    parser.add_argument("--run-evaluation", action="store_true", required=True)
    parser.add_argument(
        "--data-dir", type=Path, default=PROJECT_ROOT / "data/real_vs_fake"
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=PROJECT_ROOT / "outputs/checkpoints/best_model_v3_robust.pth",
    )
    parser.add_argument("--device", choices=["cuda", "cpu", "auto"], default="cuda")
    args = parser.parse_args()
    run_evaluation(
        args.data_dir,
        args.checkpoint,
        PROJECT_ROOT / "outputs/metrics/robustness_v2.json",
        PROJECT_ROOT / "outputs/metrics/final_test_metrics_v2.json",
        PROJECT_ROOT / "outputs",
        PROJECT_ROOT / "docs/EXPERIMENTS.md",
        args.device,
    )


if __name__ == "__main__":
    main()
