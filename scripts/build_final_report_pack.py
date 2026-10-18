"""Mevcut JSON/log/Grad-CAM dosyalarından final görselleri; model çalıştırmaz."""

import hashlib
import json
import math
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONDITIONS = {
    "CLEAN": "CLEAN",
    "JPEG_Q70": "JPEG Q70",
    "JPEG_Q50": "JPEG Q50",
    "BLUR_SIGMA1": "Blur sigma 1",
    "BLUR_SIGMA2": "Blur sigma 2",
    "NOISE_STD0.01": "Noise 0.01",
    "NOISE_STD0.03": "Noise 0.03",
    "RESIZE_112": "Resize\n224→112→224",
}
COLORS = ("#64748b", "#2563eb", "#ea580c")


def validate_metrics(metrics):
    """Grafiğe geçmeden label sırası, confusion matrix ve metrikleri doğrular."""
    if (
        metrics["class_names"] != ["REAL", "FAKE"]
        or metrics["positive_class"] != "FAKE"
    ):
        raise ValueError("Metrik label mapping REAL=0 / FAKE=1 olmalı.")
    cm = metrics["confusion_matrix"]
    if (
        len(cm) != 2
        or any(len(row) != 2 for row in cm)
        or any(type(value) is not int or value < 0 for row in cm for value in row)
    ):
        raise ValueError(
            "Confusion matrix iki sınıflı, negatif olmayan tam sayılardan oluşmalı."
        )
    tn, fp, fn, tp = cm[0] + cm[1]
    total = tn + fp + fn + tp
    if total == 0 or tp + fp == 0 or tp + fn == 0:
        raise ValueError("Metrik paydaları pozitif olmalı.")
    expected = {
        "accuracy": (tn + tp) / total,
        "precision": tp / (tp + fp),
        "recall": tp / (tp + fn),
        "f1": 2 * tp / (2 * tp + fp + fn),
    }
    for key, value in expected.items():
        if not math.isclose(metrics[key], value, rel_tol=0, abs_tol=1e-12):
            raise ValueError(f"{key} confusion matrix ile tutarsız.")
    if not math.isfinite(metrics["roc_auc"]) or not 0 <= metrics["roc_auc"] <= 1:
        raise ValueError("ROC-AUC [0,1] aralığında sonlu olmalı.")
    if metrics.get("test_image_count", total) != total:
        raise ValueError("Test görüntü sayısı confusion matrix ile tutarsız.")
    if metrics.get("correct_predictions", tn + tp) != tn + tp:
        raise ValueError("Doğru tahmin sayısı confusion matrix ile tutarsız.")
    return total


def save_figure(fig, directory, name):
    fig.savefig(directory / f"{name}.png", dpi=300, facecolor="white")
    fig.savefig(directory / f"{name}.svg", facecolor="white")
    plt.close(fig)


def build_pack(root=PROJECT_ROOT):
    root = Path(root)
    output = root / "outputs/figures/final"
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(
            "Final paket klasörü dolu; mevcut çıktılar üzerine yazılmaz."
        )
    source_paths = []

    def read_json(relative):
        path = root / relative
        source_paths.append(path)
        return json.loads(path.read_text(encoding="utf-8"))

    metrics = [
        read_json(f"outputs/metrics/{name}.json")
        for name in (
            "final_test_metrics",
            "final_test_metrics_v2",
            "final_test_metrics_v3",
        )
    ]
    for metric in metrics:
        validate_metrics(metric)
    v2, v3 = metrics[1:]
    if v2["test_fingerprint"] != v3["test_fingerprint"]:
        raise ValueError("V2 ve V3 aynı test fotoğraflarında ölçülmüş olmalı.")
    robustness = [
        read_json(f"outputs/metrics/robustness_v{version}.json") for version in (2, 3)
    ]
    for report, metric in zip(robustness, metrics[1:], strict=True):
        if (
            report["checkpoint_sha256"] != metric["checkpoint_sha256"]
            or report["test_fingerprint"] != metric["test_fingerprint"]
        ):
            raise ValueError(
                "Robustness checkpoint/test kaynağı final testle tutarsız."
            )
        for condition in CONDITIONS:
            if validate_metrics(report["results"][condition]) != 20000:
                raise ValueError("Robustness koşulunda 20.000 görüntü bulunmalı.")
        for key in (
            "accuracy",
            "precision",
            "recall",
            "f1",
            "roc_auc",
            "confusion_matrix",
        ):
            if report["results"]["CLEAN"][key] != metric[key]:
                raise ValueError("Robustness CLEAN ve final test sonuçları tutarsız.")
    checkpoint = root / "outputs/checkpoints/best_model_v3_robust.pth"
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != v3["checkpoint_sha256"]:
        raise ValueError("Final V3 checkpoint SHA-256 eşleşmiyor.")
    source_paths.append(checkpoint)
    selection_paths = list(
        (root / "outputs/figures/gradcam_v3").glob("*/selection_summary.json")
    )
    if len(selection_paths) != 1:
        raise ValueError("Tek bir doğrulanmış V3 Grad-CAM seçim kaydı bulunmalı.")
    selection = read_json(selection_paths[0].relative_to(root))
    if (
        selection["checkpoint_sha256"] != v3["checkpoint_sha256"]
        or selection["test_fingerprint"] != v3["test_fingerprint"]
    ):
        raise ValueError("Grad-CAM örnekleri final V3 checkpoint/test ile eşleşmiyor.")
    selected = {
        label: [
            e
            for e in selection["examples"]
            if e["true_label"] == e["label"] == label and not e["heatmap_empty"]
        ][:2]
        for label in ("REAL", "FAKE")
    }
    if any(len(examples) != 2 for examples in selected.values()):
        raise ValueError("Her sınıftan iki doğru, boş olmayan Grad-CAM örneği gerekli.")
    histories = []
    for relative in (
        "outputs/metrics/history.csv",
        "outputs/metrics/image_v2/history.csv",
        "outputs/metrics/image_v3_robust/history.csv",
    ):
        path = root / relative
        source_paths.append(path)
        history = pd.read_csv(path)
        required = (
            "epoch",
            "train_loss",
            "val_accuracy",
            "val_precision",
            "val_recall",
            "val_f1",
        )
        if not all(key in history and history[key].notna().all() for key in required):
            raise ValueError("Eğitim logunda gerekli metrikler eksik.")
        histories.append(history)

    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 12,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    cm = np.asarray(v3["confusion_matrix"])
    ax.imshow(cm, cmap="Blues", vmin=0, vmax=10000)
    for row in range(2):
        for column in range(2):
            ax.text(
                column,
                row,
                f"{cm[row, column]:,}".replace(",", "."),
                ha="center",
                va="center",
                fontsize=24,
                color="white" if cm[row, column] > 5000 else "#0f172a",
            )
    ax.set(
        xticks=[0, 1],
        yticks=[0, 1],
        xticklabels=["REAL", "FAKE"],
        yticklabels=["REAL", "FAKE"],
        xlabel="Tahmin edilen sınıf",
        ylabel="Gerçek sınıf",
        title="V3 Robust EfficientNet-B0 · Confusion matrix",
    )
    fig.text(
        0.5,
        0.035,
        "140k Real and Fake Faces · 20.000 test · Accuracy %99,635\n19.927 doğru / 73 yanlış · Best epoch 5",
        ha="center",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0.10, 1, 1))
    save_figure(fig, output, "v3_confusion_matrix")

    fig, axes = plt.subplots(1, 2, figsize=(15.5, 6.6))
    for ax, key, title in zip(
        axes, ("accuracy", "f1"), ("Accuracy", "F1 (FAKE)"), strict=True
    ):
        for report, label, color in zip(
            robustness, ("V2", "V3"), COLORS[1:], strict=True
        ):
            ax.plot(
                range(8),
                [100 * report["results"][condition][key] for condition in CONDITIONS],
                marker="o",
                linewidth=2.5,
                markersize=7,
                label=label,
                color=color,
            )
        ax.set(
            xticks=range(8),
            xticklabels=list(CONDITIONS.values()),
            ylim=(-2, 105),
            ylabel="Skor (%)",
            title=title,
        )
        ax.tick_params(axis="x", labelrotation=32, labelsize=10)
        ax.grid(axis="y", alpha=0.2)
        ax.legend(loc="lower left")
    fig.suptitle("V2 → V3 · Aynı 20.000 fotoğrafta kalite dayanıklılığı", fontsize=18)
    fig.text(
        0.5,
        0.025,
        "V2 gözlemleri V3 augmentation tasarımına yön verdi; geliştirme boyunca görülmemiş bağımsız test değildir.",
        ha="center",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0.09, 1, 0.96))
    save_figure(fig, output, "robustness_v2_vs_v3")

    fig, axes = plt.subplots(
        1, 2, figsize=(15.5, 6.5), gridspec_kw={"width_ratios": [2, 1]}
    )
    keys = ("accuracy", "precision", "recall", "f1")
    for index, (metric, label, color) in enumerate(
        zip(metrics, ("V1", "V2", "V3"), COLORS, strict=True)
    ):
        bars = axes[0].bar(
            np.arange(4) + (index - 1) * 0.25,
            [100 * metric[key] for key in keys],
            width=0.24,
            label=label,
            color=color,
        )
        axes[0].bar_label(bars, fmt="%.3f", rotation=90, padding=4, fontsize=9)
    axes[0].set(
        xticks=range(4),
        xticklabels=["Accuracy", "Precision\n(FAKE)", "Recall\n(FAKE)", "F1\n(FAKE)"],
        ylim=(0, 115),
        ylabel="Skor (%)",
    )
    axes[0].legend(loc="lower right")
    bars = axes[1].bar(
        ["V1", "V2", "V3"],
        [metric["roc_auc"] for metric in metrics],
        color=COLORS,
        width=0.55,
    )
    axes[1].bar_label(bars, fmt="%.6f", padding=5, fontsize=11)
    axes[1].set(ylim=(0, 1.12), ylabel="ROC-AUC (FAKE)")
    for ax in axes:
        ax.grid(axis="y", alpha=0.2)
        ax.set_axisbelow(True)
    fig.suptitle("V1 / V2 / V3 · Kendi final test sonuçları", fontsize=18)
    fig.text(
        0.5,
        0.035,
        "V1: ilk dataset, n=1.934 · V2/V3: 140k Real and Fake Faces, aynı n=20.000\nV1 farklı test setindedir; bu grafik ortak bir benchmark veya nedensel model üstünlüğü ölçümü değildir.",
        ha="center",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0.11, 1, 0.95))
    save_figure(fig, output, "model_comparison_v1_v2_v3")

    for version, (history, metric) in enumerate(
        zip(histories, metrics, strict=True), 1
    ):
        fig, axes = plt.subplots(1, 2, figsize=(14, 5.8))
        axes[0].plot(
            history["epoch"], history["train_loss"], "o-", color=COLORS[version - 1]
        )
        axes[0].set(xlabel="Epoch", ylabel="Loss", title="Eğitim loss")
        for key, label in zip(
            ("val_accuracy", "val_precision", "val_recall", "val_f1"),
            ("Accuracy", "Precision FAKE", "Recall FAKE", "F1 FAKE"),
            strict=True,
        ):
            axes[1].plot(history["epoch"], 100 * history[key], "o-", label=label)
        axes[1].set(
            xlabel="Epoch", ylabel="Skor (%)", title="CLEAN validation · Yakın görünüm"
        )
        axes[1].legend(fontsize=10)
        for ax in axes:
            ax.set_xticks(history["epoch"])
            ax.axvline(
                metric["best_epoch"],
                color="#334155",
                linestyle="--",
                alpha=0.7,
                label="Best epoch",
            )
            ax.grid(alpha=0.2)
        fig.suptitle(
            f"V{version} · Mevcut eğitim logu · Best epoch {metric['best_epoch']}",
            fontsize=17,
        )
        fig.text(
            0.5,
            0.025,
            "Train accuracy ve validation loss loglarda bulunmadığından çizilmedi; yeni eğitim yapılmadı.",
            ha="center",
            fontsize=10,
        )
        fig.tight_layout(rect=(0, 0.07, 1, 0.96))
        save_figure(fig, output, f"training_validation_v{version}")

    examples_manifest = []
    for label, examples in selected.items():
        for number, example in enumerate(examples, 1):
            # Eski mutlak dosya yolunun repository içindeki karşılığını kullan.
            source = (
                root
                / "outputs/figures/gradcam_v3"
                / Path(example["files"]["comparison"]).parent.name
                / "comparison.png"
            )
            source_paths.append(source)
            name = f"gradcam_{label.lower()}_{number:02d}_{Path(example['image']).stem}.png"
            shutil.copyfile(source, output / name)
            examples_manifest.append(
                {
                    "file": name,
                    "true_label": label,
                    "prediction": example["label"],
                    "confidence_percent": example["confidence_percent"],
                    "source": source.relative_to(root).as_posix(),
                }
            )
    manifest = {
        "purpose": "Final presentation/report pack; existing results only, no training/inference/tuning.",
        "main_checkpoint_sha256": v3["checkpoint_sha256"],
        "label_mapping": {"0": "REAL", "1": "FAKE"},
        "gradcam_selection": "Önceki seed-42 doğru örnek sırasından ilk iki REAL ve ilk iki FAKE; confidence sıralaması yapılmadı.",
        "gradcam_examples": examples_manifest,
        "source_sha256": {
            path.relative_to(root).as_posix(): hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
            for path in source_paths
        },
        "file_sha256": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(output.iterdir())
            if path.is_file()
        },
    }
    (output / "package_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Final paket: {output}; 10 PNG, 6 SVG, kaynak/hash manifesti.")


if __name__ == "__main__":
    build_pack()
