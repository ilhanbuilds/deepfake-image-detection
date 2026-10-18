"""Video-level metrics and explicit failed-detection coverage; fixed threshold 0.5."""

import argparse
import json
from pathlib import Path
from uuid import uuid4

import torch
from torch import nn
from sklearn.metrics import balanced_accuracy_score, precision_recall_fscore_support

from src.evaluate import calculate_metrics, save_metrics
from src.model import resolve_device
from src.video_manifest import DEFAULT_MANIFEST, load_manifest
from src.video_model import VideoClassifier
from src.video_pipeline import (
    FacePipeline,
    VideoDataset,
    VideoSettings,
    feature_batch,
    make_video_loader,
)


def coverage(reports: list[dict]) -> dict:
    return dict(
        attempted_videos=len(reports),
        evaluated_videos=sum(r["status"] == "accepted" for r in reports),
        rejected_videos=[r for r in reports if r["status"] != "accepted"],
        videos_with_failed_frames=[r for r in reports if r["failed_frames"]],
        failed_frame_count=sum(len(r["failed_frames"]) for r in reports),
        video_reports=reports,
    )


def calculate_video_metrics(
    targets: list[int], predictions: list[int], scores: list[float]
) -> dict:
    metrics = calculate_metrics(targets, predictions, scores)
    precision, recall, f1, support = precision_recall_fscore_support(
        targets, predictions, labels=[0, 1], zero_division=0
    )
    metrics.update(
        balanced_accuracy=float(balanced_accuracy_score(targets, predictions)),
        macro_f1=float(f1.mean()),
        per_class={
            name: dict(
                precision=float(precision[i]),
                recall=float(recall[i]),
                f1=float(f1[i]),
                support=int(support[i]),
            )
            for i, name in enumerate(("REAL", "FAKE"))
        },
    )
    return metrics


@torch.no_grad()
def evaluate_video(
    model, loader, device, cached: bool = False, amp: bool = False
) -> dict:
    model.eval()
    targets, scores, reports, predictions = [], [], [], []
    rows = []
    loss_sum = 0.0
    for batch in loader:
        reports.extend(batch["reports"])
        if not batch["items"]:
            continue
        features, lengths = feature_batch(model, batch, device, cached, amp)
        with torch.amp.autocast(device.type, enabled=amp):
            logits = model(features, lengths)
        labels = batch["labels"].to(device)
        loss = nn.functional.binary_cross_entropy_with_logits(
            logits.float(), labels, reduction="sum"
        )
        if not torch.isfinite(loss):
            raise RuntimeError("Video validation loss NaN/Inf.")
        loss_sum += loss.item()
        probabilities = logits.float().sigmoid().cpu().tolist()
        targets.extend(int(label) for label in batch["labels"].tolist())
        scores.extend(probabilities)
        predictions.extend(int(score >= 0.5) for score in probabilities)
        rows.extend(
            dict(
                path=item["record"]["path"],
                target=item["record"]["label"],
                fake_probability=score,
                prediction=int(score >= 0.5),
            )
            for item, score in zip(batch["items"], probabilities, strict=True)
        )
    metrics = calculate_video_metrics(targets, predictions, scores) if targets else None
    return dict(
        metrics=metrics,
        video_loss=loss_sum / len(targets) if targets else None,
        threshold=0.5,
        coverage=coverage(reports),
        predictions=rows,
    )


def load_video_checkpoint(path: Path, device: torch.device):
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if saved.get("pipeline") != "video_v1_frozen_gru" or saved.get("class_names") != [
        "REAL",
        "FAKE",
    ]:
        raise ValueError(
            "Video V1 checkpoint gerekli; fotoğraf checkpoint'i kabul edilmez."
        )
    model = VideoClassifier(
        pretrained=False, hidden_size=saved["config"]["hidden_size"]
    )
    model.load_state_dict(saved["model_state_dict"])
    return model.to(device).eval(), saved


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Açık final-test modu; test model seçimi/tuning için kullanılmaz."
    )
    parser.add_argument("--final-test", action="store_true", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--batch-size", type=int, choices=(1, 2), default=1)
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="auto")
    args = parser.parse_args()
    manifest = load_manifest(args.manifest)
    device = resolve_device(args.device)
    model, saved = load_video_checkpoint(args.checkpoint, device)
    if (
        saved["manifest_fingerprint"] != manifest["fingerprint"]
        or saved["epoch"] != saved["best_epoch"]
    ):
        raise ValueError(
            "Validation ile seçilmiş best checkpoint ve aynı manifest gerekli."
        )
    settings = VideoSettings(**saved["config"]["settings"])
    if FacePipeline(settings).signature != saved["config"]["pipeline"]:
        raise ValueError("Checkpoint yüz/decoder pipeline mevcut ortamla uyuşmuyor.")
    dataset = VideoDataset(
        Path(manifest["dataset_root"]), manifest["splits"]["test"], settings
    )
    loader = make_video_loader(dataset, args.batch_size)
    try:
        result = evaluate_video(model, loader, device, amp=device.type == "cuda")
    except torch.cuda.OutOfMemoryError as error:
        torch.cuda.empty_cache()
        raise RuntimeError(
            "CUDA OOM: final-test durduruldu; batch 1 / CPU kullanın."
        ) from error
    directory = Path("outputs/video_v1/evaluations") / uuid4().hex
    directory.mkdir(parents=True, exist_ok=False)
    result.update(
        checkpoint=str(args.checkpoint.resolve()),
        manifest_fingerprint=manifest["fingerprint"],
        split="official_test",
        source_identity_verified=False,
    )
    (directory / "evaluation.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if result["metrics"] is not None:
        save_metrics(result["metrics"], directory, "video_test")
    print(
        json.dumps(
            dict(
                output=str(directory),
                metrics=result["metrics"],
                attempted=result["coverage"]["attempted_videos"],
                evaluated=result["coverage"]["evaluated_videos"],
            ),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
