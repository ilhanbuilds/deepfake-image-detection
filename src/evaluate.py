"""Kaydedilmiş test ayrımında metrikler ve confusion matrix."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from torch import nn
from torch.utils.data import DataLoader

from src.dataset import (
    CLASS_NAMES,
    FaceDataset,
    audit_dataset,
    discover_samples,
    make_loader,
)
from src.model import load_checkpoint, resolve_device


def calculate_metrics(
    targets: list[int], predictions: list[int], fake_scores: list[float] | None = None
) -> dict:
    if not targets or len(targets) != len(predictions):
        raise ValueError(
            "Hedef ve tahmin listeleri boş olmamalı ve eşit uzunlukta olmalı."
        )
    if not set(targets + predictions).issubset({0, 1}):
        raise ValueError("Etiketler yalnızca 0 (REAL) ve 1 (FAKE) olabilir.")
    precision, recall, f1, _ = precision_recall_fscore_support(
        targets, predictions, average="binary", pos_label=1, zero_division=0
    )
    metrics = {
        "accuracy": float(accuracy_score(targets, predictions)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "positive_class": "FAKE",
        "class_names": list(CLASS_NAMES),
        "confusion_matrix": confusion_matrix(
            targets, predictions, labels=[0, 1]
        ).tolist(),
    }
    if fake_scores is not None:
        if len(fake_scores) != len(targets) or not all(
            np.isfinite(score) and 0 <= score <= 1 for score in fake_scores
        ):
            raise ValueError(
                "FAKE olasılıkları hedeflerle eşit uzunlukta, sonlu ve [0, 1] aralığında olmalı."
            )
        metrics["roc_auc"] = (
            float(roc_auc_score(targets, fake_scores))
            if len(set(targets)) == 2
            else None
        )
    return metrics


@torch.inference_mode()
def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    include_roc_auc: bool = False,
) -> dict:
    model.eval()
    targets: list[int] = []
    predictions: list[int] = []
    fake_scores: list[float] = []
    for images, labels in loader:
        logits = model(images.to(device, non_blocking=True))
        targets.extend(labels.tolist())
        predictions.extend(logits.argmax(dim=1).cpu().tolist())
        if include_roc_auc:
            fake_scores.extend(torch.softmax(logits, dim=1)[:, 1].cpu().tolist())
    return calculate_metrics(
        targets, predictions, fake_scores if include_roc_auc else None
    )


def save_metrics(metrics: dict, output_dir: str | Path, name: str = "test") -> None:
    output_dir = Path(output_dir)
    metrics_dir, figures_dir = output_dir / "metrics", output_dir / "figures"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    (metrics_dir / f"{name}.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    matrix = np.asarray(metrics["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.imshow(matrix, cmap="Blues")
    ax.set(
        xticks=[0, 1],
        yticks=[0, 1],
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        xlabel="Tahmin",
        ylabel="Gerçek",
        title="Confusion matrix",
    )
    for row in range(2):
        for col in range(2):
            ax.text(
                col,
                row,
                str(matrix[row, col]),
                ha="center",
                va="center",
                color="white" if matrix[row, col] > matrix.max() / 2 else "black",
            )
    fig.tight_layout()
    fig.savefig(figures_dir / f"{name}_confusion_matrix.png", dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Checkpoint'in ayrılmış test verisini değerlendirir."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument(
        "--name", choices=["test", "final_test_metrics"], default="test"
    )
    args = parser.parse_args()
    device = resolve_device(args.device)
    model, checkpoint = load_checkpoint(args.checkpoint, device)
    splits = checkpoint["splits"]
    memberships = {}
    for name, subset in splits.items():
        for path, _ in subset:
            if path in memberships:
                raise ValueError(
                    "Checkpoint ayrımlarında yinelenen veya çakışan dosya yolu var."
                )
            memberships[path] = name
    expected = sorted(tuple(sample) for subset in splits.values() for sample in subset)
    if sorted(discover_samples(args.data_dir)) != expected:
        raise ValueError(
            "Dataset dosya listesi checkpoint ile eşleşmiyor; orijinal dataset'i kullanın."
        )
    if "dataset_fingerprint" in checkpoint:
        audit = audit_dataset(args.data_dir, checkpoint["seed"], progress=True)
        if (
            audit["fingerprint"] != checkpoint["dataset_fingerprint"]
            or audit["errors"]
            or audit["path_errors"]
        ):
            raise ValueError("Dataset içeriği checkpoint ile eşleşmiyor.")
        group_memberships = {}
        for path, name in memberships.items():
            group = audit["groups"][path]
            if group in group_memberships and group_memberships[group] != name:
                raise ValueError(
                    "Checkpoint ayrımlarında bilinen kopya/augmentation grubu sızıntısı var."
                )
            group_memberships[group] = name
    loader = make_loader(
        FaceDataset(args.data_dir, splits["test"]),
        args.batch_size,
        args.workers,
        pin_memory=device.type == "cuda",
    )
    metrics = evaluate_model(model, loader, device, include_roc_auc=True)
    metrics.update(
        best_epoch=checkpoint["epoch"],
        split="test",
        test_image_count=len(splits["test"]),
        test_class_counts={
            name: sum(label == index for _, label in splits["test"])
            for index, name in enumerate(CLASS_NAMES)
        },
    )
    if "dataset_fingerprint" in checkpoint:
        metrics.update(
            dataset_fingerprint=checkpoint["dataset_fingerprint"],
            source_identity_verified=audit["source_identity_verified"],
            source_limitation=audit["source_limitation"],
        )
    save_metrics(metrics, args.output_dir, args.name)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
