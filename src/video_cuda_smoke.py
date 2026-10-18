"""Bounded real-cache CUDA updates; no epoch, checkpoint, validation or test inference."""

import argparse
import json
from pathlib import Path
from time import perf_counter
from uuid import uuid4

import torch

from src.dataset import seed_everything
from src.model import resolve_device
from src.video_cache import audit_feature_cache
from src.video_manifest import DEFAULT_MANIFEST, load_manifest
from src.video_model import VideoClassifier
from src.video_pipeline import FacePipeline, FeatureCache, make_video_loader
from src.video_train import train_video_batch, training_class_balance


def select_smoke_items(records, cache, root):
    selected = {}
    for record in records:
        item = cache.read(record, root)
        if item["values"] is None:
            continue
        key = (len(item["values"]), record["label"])
        if key not in selected:
            selected[key] = item
        if len(selected) == 8:
            break
    if set(selected) != {(length, label) for length in range(3, 7) for label in (0, 1)}:
        raise ValueError(
            "CUDA smoke için iki sınıfta 3–6 geçerli kare örnekleri gerekli."
        )
    # REAL-only and FAKE-only pairs exercise the minority-weight overflow trigger.
    return [selected[(length, label)] for label in (0, 1) for length in range(3, 7)]


def main():
    parser = argparse.ArgumentParser(description="Yalnız 4 CUDA batch; tam eğitim yok.")
    parser.add_argument("--smoke", required=True, action="store_true")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--cache-dir", type=Path, default=Path("outputs/video_v1/cache")
    )
    parser.add_argument("--precision", choices=("fp32", "amp"), default="fp32")
    args = parser.parse_args()
    device = resolve_device("cuda")
    seed_everything(42)
    model = VideoClassifier(pretrained=True).to(device).train()
    manifest = load_manifest(args.manifest)
    cache = FeatureCache(
        args.cache_dir, FacePipeline().signature, model.backbone_fingerprint()
    )
    audit = audit_feature_cache(manifest, cache)
    records = audit["splits"]["train"]["accepted_records"]
    balance = training_class_balance(records)
    weights = torch.tensor(
        [balance["class_weights"][name] for name in ("REAL", "FAKE")], device=device
    )
    items = select_smoke_items(records, cache, Path(manifest["dataset_root"]))
    before_head = model.head.weight.detach().clone()
    before_backbone = model.backbone_fingerprint()
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=3e-4
    )
    amp = args.precision == "amp"
    scaler = torch.amp.GradScaler("cuda", enabled=amp, init_scale=1024.0)
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    started, rows = perf_counter(), []
    for index, batch in enumerate(make_video_loader(items, batch_size=2), 1):
        result = train_video_batch(
            model, batch, device, optimizer, scaler, True, amp, weights
        )
        if not all(torch.isfinite(p).all() for p in model.parameters()):
            raise RuntimeError("Smoke optimizer sonrası model parametresi NaN/Inf.")
        row = dict(batch=index, **result)
        row["logits"] = result["logits"].cpu().tolist()
        row["paths"] = [item["record"]["path"] for item in batch["items"]]
        row["labels"] = batch["labels"].tolist()
        rows.append(row)
    torch.cuda.synchronize(device)
    if (
        torch.equal(before_head, model.head.weight)
        or before_backbone != model.backbone_fingerprint()
    ):
        raise RuntimeError("GRU/head güncellenmeli, frozen backbone değişmemeli.")
    report = dict(
        schema="video_cuda_update_smoke_v1",
        precision=args.precision,
        device=torch.cuda.get_device_name(device),
        batches=rows,
        optimizer_steps=len(rows),
        elapsed_seconds=perf_counter() - started,
        peak_cuda_allocated_mib=torch.cuda.max_memory_allocated(device) / 1024**2,
        peak_cuda_reserved_mib=torch.cuda.max_memory_reserved(device) / 1024**2,
        class_balance=balance,
        cache_fingerprint=audit["cache_fingerprint"],
        official_test_accessed=False,
        validation_updates=False,
        backbone_unchanged=True,
        head_updated=True,
        checkpoint_written=False,
    )
    directory = Path("outputs/video_v1/cuda_smoke") / uuid4().hex
    directory.mkdir(parents=True, exist_ok=False)
    (directory / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"CUDA smoke raporu: {directory / 'report.json'}")


if __name__ == "__main__":
    main()
