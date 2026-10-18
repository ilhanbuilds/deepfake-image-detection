"""Single video inference via the exact shared crop pipeline; no silent fallback."""

import argparse
import json
from pathlib import Path

import torch

from src.model import resolve_device
from src.video_evaluate import load_video_checkpoint
from src.video_pipeline import FacePipeline, VideoSettings


@torch.no_grad()
def predict_video(
    model, path: Path, device: torch.device, pipeline: FacePipeline
) -> dict:
    model.eval()
    frames, report, _ = pipeline.read(path)
    if frames is None:
        return dict(prediction=None, fake_probability=None, detection_report=report)
    features = model.extract_features(frames.to(device))
    logits = model(features.unsqueeze(0), torch.tensor([len(features)])).float()
    if not torch.isfinite(features).all() or not torch.isfinite(logits).all():
        raise RuntimeError("Video inference özellik/logit NaN/Inf; tahmin üretilmedi.")
    score = logits.sigmoid().item()
    return dict(
        prediction="FAKE" if score >= 0.5 else "REAL",
        fake_probability=score,
        threshold=0.5,
        detection_report=report,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Eğitilmiş ayrı Video V1 checkpoint ile video inference."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="auto")
    args = parser.parse_args()
    device = resolve_device(args.device)
    model, saved = load_video_checkpoint(args.checkpoint, device)
    pipeline = FacePipeline(VideoSettings(**saved["config"]["settings"]))
    if pipeline.signature != saved["config"]["pipeline"]:
        raise ValueError("Checkpoint yüz/decoder pipeline mevcut ortamla uyuşmuyor.")
    try:
        result = predict_video(model, args.video, device, pipeline)
    except torch.cuda.OutOfMemoryError as error:
        torch.cuda.empty_cache()
        raise RuntimeError(
            "CUDA OOM: video inference durdu; --device cpu kullanın."
        ) from error
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
