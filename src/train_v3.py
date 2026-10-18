"""Ayrı V3 robust fotoğraf eğitimi; yalnızca TRAIN augmentation, CLEAN validation."""

import argparse
import json
import math
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from time import perf_counter
from uuid import uuid4

import pandas as pd
import torch
from torch import nn

from src.dataset import CLASS_NAMES, IMAGE_SIZE, make_loader, seed_everything
from src.evaluate import evaluate_model
from src.experiment_log import append_experiment
from src.model import build_model, resolve_device
from src.presplit_dataset import (
    dataset_fingerprint,
    resolve_data_root,
    split_counts,
)
from src.robust_augmentation import AugmentationSettings, RobustPresplitDataset
from src.train import (
    EarlyStopping,
    atomic_copy,
    atomic_save,
    calculate_timing,
    capture_rng,
    restore_rng,
    select_resume,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def validate_v3_resume(checkpoint, fingerprint, config, data_root):
    if (
        checkpoint.get("pipeline") != "presplit_image_v3_robust"
        or checkpoint.get("dataset_fingerprint") != fingerprint
        or checkpoint.get("config") != config
        or checkpoint.get("dataset_root") != str(data_root)
        or checkpoint.get("class_names") != list(CLASS_NAMES)
        or checkpoint.get("image_size") != IMAGE_SIZE
    ):
        raise ValueError("v3 resume model/dataset/config uyumsuz.")
    required = {
        "optimizer_state_dict",
        "scaler_state_dict",
        "rng_state",
        "early_stopping",
        "history",
        "best_epoch",
    }
    if not required.issubset(checkpoint):
        raise ValueError("Tam resume durumu eksik.")


def make_parser():
    parser = argparse.ArgumentParser(
        description="V3: yalnızca açık komutla eğitim; train kalite augmentation, CLEAN valid."
    )
    parser.add_argument("--run-training", action="store_true", required=True)
    parser.add_argument(
        "--data-dir", type=Path, default=PROJECT_ROOT / "data/real_vs_fake"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=6,
        help="Resume dahil toplam epoch; benchmark için 1.",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, choices=[0], default=0)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["cuda"], default="cuda")
    parser.add_argument("--resume", nargs="?", const="auto")
    return parser


def run_training(args, record):
    seed_everything(args.seed)
    device = resolve_device(args.device)
    data_root = resolve_data_root(args.data_dir)
    settings = AugmentationSettings()
    datasets = {
        name: RobustPresplitDataset(data_root, name, settings)
        for name in ("train", "valid")
    }
    record["counts"] = {**split_counts(datasets), "test": "erişilmedi"}
    print(
        json.dumps(
            {
                "class_to_idx": {"REAL": 0, "FAKE": 1},
                "counts": record["counts"],
                "augmentation": asdict(settings),
            },
            indent=2,
        ),
        flush=True,
    )
    amp_enabled = True
    config = {
        "batch_size": args.batch_size,
        "workers": args.workers,
        "learning_rate": args.learning_rate,
        "seed": args.seed,
        "amp": amp_enabled,
        "device_type": device.type,
        "patience": args.patience,
        "min_delta": 0.0,
        "optimizer": "AdamW",
        "augmentation": asdict(settings),
        "augmentation_version": 1,
        "blur_engine": "Pillow.GaussianBlur",
    }
    args.min_delta = 0.0
    record["settings"] = {
        **config,
        "image_size": IMAGE_SIZE,
        "epochs": args.epochs,
        "preprocessing": "EXIF/RGB -> 224 -> TRAIN quality augmentation -> ImageNet normalization; valid CLEAN",
    }
    checkpoint_dir = PROJECT_ROOT / "outputs/checkpoints/image_v3_robust"
    metrics_dir = PROJECT_ROOT / "outputs/metrics/image_v3_robust"
    best_path = PROJECT_ROOT / "outputs/checkpoints/best_model_v3_robust.pth"
    last_path = checkpoint_dir / "last.pth"
    resume_path = select_resume(checkpoint_dir, args.resume)
    if resume_path is None and (
        best_path.exists()
        or (checkpoint_dir.exists() and any(checkpoint_dir.glob("*.pth")))
        or (metrics_dir / "history.csv").exists()
    ):
        raise FileExistsError("v3 checkpoint mevcut; --resume auto kullanın.")
    record["files"] = [str(best_path), str(last_path), str(metrics_dir / "history.csv")]
    fingerprint = dataset_fingerprint(datasets)
    checkpoint = None
    best_metrics = {}
    if resume_path is not None:
        resume_path = resume_path.resolve()
        if resume_path.parent != checkpoint_dir.resolve():
            raise ValueError(
                "Resume yalnızca image_v3_robust epoch dizininden yapılır."
            )
        checkpoint = torch.load(resume_path, map_location="cpu", weights_only=True)
        validate_v3_resume(checkpoint, fingerprint, config, data_root)
        best_epoch_path = checkpoint_dir / f"epoch_{checkpoint['best_epoch']:03d}.pth"
        if not best_epoch_path.exists():
            raise FileNotFoundError("En iyi v3 epoch eksik.")
        best_checkpoint = torch.load(
            best_epoch_path, map_location="cpu", weights_only=True
        )
        validate_v3_resume(best_checkpoint, fingerprint, config, data_root)
        if (
            best_checkpoint["epoch"] != checkpoint["best_epoch"]
            or best_checkpoint["val_metrics"]["f1"] != checkpoint["best_f1"]
        ):
            raise ValueError("En iyi v3 epoch durumu uyumsuz.")
        best_metrics = best_checkpoint["val_metrics"]
        del best_checkpoint
        atomic_copy(best_epoch_path, best_path)
    train_loader = make_loader(
        datasets["train"],
        args.batch_size,
        args.workers,
        shuffle=True,
        pin_memory=device.type == "cuda",
        seed=args.seed,
    )
    val_loader = make_loader(
        datasets["valid"],
        args.batch_size,
        args.workers,
        pin_memory=device.type == "cuda",
        seed=args.seed,
    )
    model = build_model(pretrained=checkpoint is None).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    criterion = nn.CrossEntropyLoss()
    stopper = EarlyStopping(args.patience, args.min_delta)
    history, best_f1, best_epoch, start_epoch = [], -1.0, 0, 1
    if checkpoint is not None:
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
        stopper = EarlyStopping(**checkpoint["early_stopping"])
        history, best_f1 = checkpoint["history"], checkpoint["best_f1"]
        start_epoch = checkpoint["epoch"] + 1
        best_epoch = checkpoint["best_epoch"]
        restore_rng(checkpoint["rng_state"], train_loader, val_loader)
        del checkpoint
        if stopper.bad_epochs >= stopper.patience:
            print("Bu deney early stopping ile tamamlanmış. Yeni epoch başlatılmadı.")
            return {
                "history": history,
                "best_epoch": best_epoch,
                "best_validation_metrics": best_metrics,
                "new_epochs": 0,
            }
        print(f"Resume: {resume_path}, sıradaki epoch={start_epoch}", flush=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    try:
        for epoch in range(start_epoch, args.epochs + 1):
            epoch_path = checkpoint_dir / f"epoch_{epoch:03d}.pth"
            if epoch_path.exists():
                raise FileExistsError(
                    f"Daha yeni epoch korunuyor: {epoch_path}; --resume auto kullanın."
                )
            epoch_started = perf_counter()
            model.train()
            loss_sum, count = 0.0, 0
            for batch_index, (images, labels) in enumerate(train_loader, 1):
                images, labels = (
                    images.to(device, non_blocking=True),
                    labels.to(device, non_blocking=True),
                )
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
                    loss = criterion(model(images), labels)
                if not torch.isfinite(loss).item():
                    raise RuntimeError("Loss NaN/Inf; son tamamlanan epoch korunuyor.")
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                loss_sum += loss.item() * labels.size(0)
                count += labels.size(0)
                if batch_index % 100 == 0:
                    print(
                        f"Epoch {epoch}/{args.epochs}; batch {batch_index}/{len(train_loader)}; loss={loss_sum / count:.4f}",
                        flush=True,
                    )
            optimizer.zero_grad(set_to_none=True)
            metrics = evaluate_model(model, val_loader, device)
            row = {
                "epoch": epoch,
                "train_loss": loss_sum / count,
                **{
                    f"val_{key}": metrics[key]
                    for key in ("accuracy", "precision", "recall", "f1")
                },
            }
            history.append(row)
            improved = metrics["f1"] > best_f1
            best_f1 = max(best_f1, metrics["f1"])
            if improved:
                best_epoch = epoch
                best_metrics = metrics
            stopped = stopper.update(metrics["f1"])
            row.update(
                calculate_timing(
                    history[:-1],
                    perf_counter() - epoch_started,
                    0 if stopped else args.epochs - epoch,
                )
            )
            state = {
                "training_state_version": 1,
                "architecture": "efficientnet_b0",
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "class_names": list(CLASS_NAMES),
                "image_size": IMAGE_SIZE,
                "seed": args.seed,
                "epoch": epoch,
                "config": config,
                "pipeline": "presplit_image_v3_robust",
                "dataset_root": str(data_root),
                "best_epoch": best_epoch,
                "dataset_fingerprint": fingerprint,
                "val_metrics": metrics,
                "early_stopping": asdict(stopper),
                "best_f1": best_f1,
                "history": history,
                "rng_state": capture_rng(train_loader, val_loader),
            }
            atomic_save(state, epoch_path)
            if improved:
                atomic_copy(epoch_path, best_path)
            atomic_copy(epoch_path, last_path)
            pd.DataFrame(history).to_csv(metrics_dir / "history.csv", index=False)
            print(json.dumps(row), flush=True)
            print(
                f"Epoch {epoch}/{args.epochs} süre: {row['epoch_duration']:.2f} saniye "
                f"({row['epoch_duration_minutes']:.2f} dakika); "
                f"toplam eğitim: {row['total_training_duration']:.2f} saniye "
                f"({row['total_training_duration_minutes']:.2f} dakika); "
                f"ETA: {row['eta']:.2f} saniye ({row['eta_minutes']:.2f} dakika)",
                flush=True,
            )
            if stopped:
                print(
                    f"Early stopping: {args.patience} epoch boyunca validation F1 artmadı."
                )
                break
    except KeyboardInterrupt:
        print(
            f"Eğitim durduruldu. Son tamamlanan epoch: {last_path}; aynı --resume auto komutuyla devam edin."
        )
        raise SystemExit(130) from None
    print(f"En iyi model: {best_path}. Test ayrımı eğitimde kullanılmadı.")
    return {
        "history": history,
        "best_epoch": best_epoch,
        "best_validation_metrics": best_metrics,
        "new_epochs": max(0, len(history) - start_epoch + 1),
    }


def main():
    parser = make_parser()
    args = parser.parse_args()
    if (
        not 1 <= args.epochs <= 6
        or args.batch_size < 1
        or args.patience < 1
        or not math.isfinite(args.learning_rate)
        or args.learning_rate <= 0
    ):
        parser.error("Epoch 1–6; batch/LR/patience pozitif olmalı.")
    started = perf_counter()
    record = {
        "id": f"v3-robust-training-{uuid4().hex}",
        "name": "V3 Robust EfficientNet-B0 eğitim çalıştırması",
        "date": datetime.now(timezone(timedelta(hours=3))).isoformat(
            timespec="seconds"
        ),
        "model": "EfficientNet-B0 / ImageNet IMAGENET1K_V1",
        "checkpoint": str(
            PROJECT_ROOT / "outputs/checkpoints/best_model_v3_robust.pth"
        ),
        "dataset": str(args.data_dir),
        "counts": {},
        "objective": "Train kalite augmentation ile CLEAN validation FAKE F1 seçimi",
        "settings": {},
        "results": {},
        "duration_seconds": 0,
        "observations": "",
        "conclusion": "",
        "files": [],
        "status": "failed",
    }
    try:
        result = run_training(args, record)
        record.update(
            status="completed",
            results=result,
            observations="Validation CLEAN; test klasörü okunmadı.",
            conclusion="Bu çalıştırma tamamlandı; final test veya robustness sonucu üretilmedi.",
        )
    except BaseException as error:
        record.update(
            observations=f"{type(error).__name__}: {error}",
            conclusion="Çalıştırma kesildi/başarısız; tamamlanmış epoch checkpoint'leri korunur.",
        )
        raise
    finally:
        record["duration_seconds"] = perf_counter() - started
        append_experiment(PROJECT_ROOT / "docs/EXPERIMENTS.md", record)


if __name__ == "__main__":
    main()
