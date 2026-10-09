"""Açık komutla eğitim; epoch checkpoint'leri, early stopping ve güvenli resume."""

import argparse
import json
import math
import os
import random
import shutil
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import torch
from torch import nn

from src.dataset import (
    CLASS_NAMES,
    IMAGE_SIZE,
    SEED,
    FaceDataset,
    audit_dataset,
    make_loader,
    seed_everything,
)
from src.evaluate import evaluate_model
from src.model import build_model, environment_info, resolve_device


@dataclass
class EarlyStopping:
    patience: int = 3
    min_delta: float = 0.0
    best_score: float = -1.0
    bad_epochs: int = 0

    def update(self, score: float) -> bool:
        if not math.isfinite(score):
            raise ValueError("Validation F1 sonlu olmalı.")
        if score > self.best_score + self.min_delta:
            self.best_score = score
            self.bad_epochs = 0
        else:
            self.bad_epochs += 1
        return self.bad_epochs >= self.patience


def atomic_save(checkpoint: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        dir=path.parent, prefix=path.name, suffix=".tmp"
    )
    os.close(descriptor)
    try:
        torch.save(checkpoint, temporary)
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def atomic_copy(source: Path, destination: Path) -> None:
    descriptor, temporary = tempfile.mkstemp(
        dir=destination.parent, prefix=destination.name, suffix=".tmp"
    )
    os.close(descriptor)
    try:
        shutil.copyfile(source, temporary)
        os.replace(temporary, destination)
    finally:
        Path(temporary).unlink(missing_ok=True)


def capture_rng(train_loader, val_loader) -> dict:
    numpy_state = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": [
            numpy_state[0],
            numpy_state[1].tolist(),
            int(numpy_state[2]),
            int(numpy_state[3]),
            float(numpy_state[4]),
        ],
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "train_generator": train_loader.generator.get_state(),
        "val_generator": val_loader.generator.get_state(),
    }


def restore_rng(state: dict, train_loader, val_loader) -> None:
    random.setstate(state["python"])
    numpy_state = state["numpy"]
    np.random.set_state(
        (
            numpy_state[0],
            np.asarray(numpy_state[1], dtype=np.uint32),
            numpy_state[2],
            numpy_state[3],
            numpy_state[4],
        )
    )
    torch.set_rng_state(state["torch"])
    if state["cuda"] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])
    train_loader.generator.set_state(state["train_generator"])
    val_loader.generator.set_state(state["val_generator"])


def validate_resume(checkpoint: dict, audit: dict, config: dict) -> None:
    if checkpoint.get("training_state_version") != 1:
        raise ValueError("Checkpoint tam resume durumunu içermiyor.")
    if checkpoint["dataset_fingerprint"] != audit["fingerprint"]:
        raise ValueError("Dataset içeriği değişmiş; resume iptal edildi.")
    saved_splits = {
        name: [tuple(sample) for sample in subset]
        for name, subset in checkpoint["splits"].items()
    }
    if saved_splits != audit["splits"]:
        raise ValueError("Dataset ayrımı değişmiş; resume iptal edildi.")
    if checkpoint["config"] != config:
        raise ValueError(
            "Resume için batch/LR/seed/AMP/workers/early stopping ayarları aynı olmalı."
        )
    if (
        checkpoint["class_names"] != list(CLASS_NAMES)
        or checkpoint["image_size"] != IMAGE_SIZE
    ):
        raise ValueError("Checkpoint sınıf eşlemesi veya boyutu uyumsuz.")


def select_resume(checkpoint_dir: Path, requested: str | None) -> Path | None:
    if requested != "auto":
        return Path(requested) if requested else None
    # Tam yazılmış epoch dosyası, last.pth güncellemesinden önceki kesintiyi de kurtarır.
    epochs = sorted(checkpoint_dir.glob("epoch_[0-9][0-9][0-9].pth"))
    if epochs:
        return epochs[-1]
    last = checkpoint_dir / "last.pth"
    return last if last.exists() else None


def calculate_timing(
    history: list[dict], epoch_duration: float, remaining_epochs: int
) -> dict[str, float]:
    durations = [
        float(row["epoch_duration"]) for row in history if "epoch_duration" in row
    ]
    durations.append(epoch_duration)
    total = sum(durations)
    eta = total / len(durations) * max(0, remaining_epochs)
    return {
        "epoch_duration": round(epoch_duration, 6),
        "epoch_duration_minutes": round(epoch_duration / 60, 6),
        "total_training_duration": round(total, 6),
        "total_training_duration_minutes": round(total / 60, 6),
        "eta": round(eta, 6),
        "eta_minutes": round(eta / 60, 6),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Varsayılan olarak eğitim yapmaz; açık çalışma modu gerekir."
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--check-data",
        action="store_true",
        help="Decode/hash/grup/split denetimi; eğitim yapmaz.",
    )
    mode.add_argument(
        "--run-training", action="store_true", help="Gerçek eğitimi başlatır."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument(
        "--epochs", type=int, default=10, help="Resume dahil toplam epoch üst sınırı."
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--min-delta", type=float, default=0.0)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument(
        "--resume",
        nargs="?",
        const="auto",
        help="auto: son tamamlanan epoch varsa sürdürür, yoksa yeni deney.",
    )
    args = parser.parse_args()
    if (
        args.epochs < 1
        or args.batch_size < 1
        or args.workers < 0
        or args.patience < 1
        or not math.isfinite(args.learning_rate)
        or args.learning_rate <= 0
        or not math.isfinite(args.min_delta)
        or args.min_delta < 0
    ):
        parser.error(
            "Epoch/batch/LR/patience pozitif; workers/min-delta negatif olmamalı."
        )
    if args.resume and args.workers != 0:
        parser.error("Tekrarlanabilir epoch resume için --workers 0 kullanın.")
    seed_everything(args.seed)
    audit = audit_dataset(args.data_dir, args.seed, progress=True)
    print(
        json.dumps(
            {
                key: audit.get(key)
                for key in (
                    "counts",
                    "split_counts",
                    "csv",
                    "augmentation_images",
                    "cross_split_groups",
                    "source_limitation",
                )
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    if audit["errors"] or audit["path_errors"] or "split_error" in audit:
        raise ValueError(
            f"Dataset denetimi başarısız: bozuk={len(audit['errors'])}, yol={len(audit['path_errors'])}, split={audit.get('split_error')}"
        )
    if audit.get("csv", {}).get("local_mapping_matches") is False:
        raise ValueError("CSV etiket/dosya listesi yerel dataset ile eşleşmiyor.")
    if args.check_data:
        print(
            f"Decode/hash/yol/grup ayrımı doğrulandı. Kopya eşleşmesi: {len(audit['duplicates'])}. Eğitim başlatılmadı."
        )
        return
    splits = audit["splits"]
    device = resolve_device(args.device)
    amp_enabled = device.type == "cuda" and not args.no_amp
    config = {
        "batch_size": args.batch_size,
        "workers": args.workers,
        "learning_rate": args.learning_rate,
        "seed": args.seed,
        "amp": amp_enabled,
        "device_type": device.type,
        "patience": args.patience,
        "min_delta": args.min_delta,
        "optimizer": "AdamW",
    }
    print(json.dumps(environment_info(), indent=2))
    print(f"Device: {device}; AMP: {amp_enabled}; ayarlar: {config}", flush=True)
    checkpoint_dir, metrics_dir = (
        args.output_dir / "checkpoints",
        args.output_dir / "metrics",
    )
    last_path, best_path = (
        checkpoint_dir / "last.pth",
        checkpoint_dir / "best_model.pth",
    )
    resume_path = select_resume(checkpoint_dir, args.resume)
    if (
        resume_path is None
        and checkpoint_dir.exists()
        and any(checkpoint_dir.glob("*.p*"))
    ):
        raise FileExistsError(
            "Mevcut checkpoint'ler korunuyor; --resume auto veya yeni --output-dir kullanın."
        )
    checkpoint = None
    if resume_path is not None:
        if resume_path.parent.resolve() != checkpoint_dir.resolve():
            raise ValueError(
                "Resume dosyası --output-dir/checkpoints altında olmalı; aynı deney dizinini kullanın."
            )
        checkpoint = torch.load(resume_path, map_location="cpu", weights_only=True)
        validate_resume(checkpoint, audit, config)
        if checkpoint["val_metrics"]["f1"] == checkpoint["best_f1"]:
            atomic_copy(resume_path, best_path)
        elif not best_path.exists():
            raise FileNotFoundError(
                "En iyi checkpoint eksik; aynı deneyin best_model.pth dosyasını geri yükleyin."
            )
    train_loader = make_loader(
        FaceDataset(args.data_dir, splits["train"], training=True),
        args.batch_size,
        args.workers,
        shuffle=True,
        pin_memory=device.type == "cuda",
        seed=args.seed,
    )
    val_loader = make_loader(
        FaceDataset(args.data_dir, splits["val"]),
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
    history, best_f1, start_epoch = [], -1.0, 1
    if checkpoint is not None:
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
        stopper = EarlyStopping(**checkpoint["early_stopping"])
        history, best_f1 = checkpoint["history"], checkpoint["best_f1"]
        start_epoch = checkpoint["epoch"] + 1
        restore_rng(checkpoint["rng_state"], train_loader, val_loader)
        del checkpoint
        if stopper.bad_epochs >= stopper.patience:
            print("Bu deney early stopping ile tamamlanmış. Yeni epoch başlatılmadı.")
            return
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
                "splits": splits,
                "dataset_fingerprint": audit["fingerprint"],
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


if __name__ == "__main__":
    main()
