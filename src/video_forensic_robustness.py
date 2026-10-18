"""Paired, post-crop stress pilot; not an end-to-end detector robustness claim."""

import argparse
import copy
import hashlib
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from src.video_forensic import (
    ROOT,
    SBIClassifier,
    SBIFaces,
    accepted_roster,
    frame_scores,
    score_faces,
    summarize,
    write_json,
)
from src.video_pipeline import opencv
from src.video_quality_fusion import fuse

SEEDS = (42, 123, 2026)
CONDITIONS = ("clean", "blur", "jpeg", "drop_frame", "crop_translation")


def corrupt(pixels, report, condition, seed, path):
    if condition not in CONDITIONS:
        raise ValueError("Unknown corruption.")
    cv = opencv()
    pixels, report = pixels.clone(), copy.deepcopy(report)
    stable_seed = int.from_bytes(
        hashlib.sha256(f"{seed}:{path}".encode()).digest()[:8], "big"
    )
    rng = np.random.default_rng(stable_seed)
    frames = report["frames"]
    if condition == "drop_frame" and len(frames) > 3:
        del frames[int(rng.integers(len(frames)))]
    affected = set(
        rng.choice(len(frames), size=max(1, len(frames) // 2), replace=False).tolist()
    )
    for i, frame in enumerate(frames):
        for face in frame["faces"]:
            index = face["crop_index"]
            rgb = pixels[index].permute(1, 2, 0).numpy().copy()
            if i in affected:
                if condition == "blur":
                    rgb = cv.GaussianBlur(rgb, (0, 0), 1.5)
                elif condition == "jpeg":
                    ok, encoded = cv.imencode(
                        ".jpg",
                        cv.cvtColor(rgb, cv.COLOR_RGB2BGR),
                        [cv.IMWRITE_JPEG_QUALITY, 50],
                    )
                    if not ok:
                        raise RuntimeError("JPEG encode failed.")
                    rgb = cv.cvtColor(
                        cv.imdecode(encoded, cv.IMREAD_COLOR), cv.COLOR_BGR2RGB
                    )
                elif condition == "crop_translation":
                    # Synthetic translation inside the saved crop, with reflected border.
                    # This is explicitly NOT a re-run of RetinaFace on jittered video bboxes.
                    dx, dy = rng.choice([-19, 19], size=2)
                    rgb = cv.warpAffine(
                        rgb,
                        np.float32([[1, 0, dx], [0, 1, dy]]),
                        (380, 380),
                        borderMode=cv.BORDER_REFLECT_101,
                    )
            pixels[index] = torch.from_numpy(rgb.transpose(2, 0, 1).copy())
            gray = cv.cvtColor(rgb, cv.COLOR_RGB2GRAY)
            # Recompute at 380 for EVERY condition including clean: matched quality scope.
            face["sharpness"] = float(np.log1p(cv.Laplacian(gray, cv.CV_64F).var()))
    return pixels, report


def main():
    parser = argparse.ArgumentParser(
        description="36-video source-stratified paired validation stress pilot; no training."
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "FFraw.tar")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Existing pilot protected.")
    manifest, audit, records = accepted_roster("val")
    selected = [
        r
        for source in ("Celeb-real", "YouTube-real", "Celeb-synthesis")
        for r in sorted(
            [r for r in records if r["path"].startswith(source + "/")],
            key=lambda r: hashlib.sha256(r["path"].encode()).hexdigest(),
        )[:12]
    ]
    torch.set_num_threads(4)
    model = SBIClassifier(args.checkpoint).cuda().eval()
    extractor = SBIFaces("cuda")
    examples = [
        (r, *extractor.read(Path(manifest["dataset_root"]), r, audit)) for r in selected
    ]
    if any(report["status"] != "accepted" for _, _, report in examples):
        raise ValueError("Pilot cohort coverage mismatch.")
    began = perf_counter()
    conditions = {}
    for seed in SEEDS:
        for condition in CONDITIONS:
            rows = []
            for record, pixels, report in examples:
                altered, metadata = corrupt(
                    pixels, report, condition, seed, record["path"]
                )
                scores = frame_scores(score_faces(model, altered, "cuda"), metadata)
                weighted, weights = fuse(scores, metadata)
                rows.append(
                    dict(
                        path=record["path"],
                        target=record["label"],
                        status="accepted",
                        fake_probability=float(np.mean(scores)),
                        prediction=int(np.mean(scores) >= 0.5),
                        quality_probability=weighted,
                        quality_prediction=int(weighted >= 0.5),
                        weights=weights,
                    )
                )
            quality_rows = [
                dict(
                    r,
                    fake_probability=r["quality_probability"],
                    prediction=r["quality_prediction"],
                )
                for r in rows
            ]
            conditions[f"{seed}:{condition}"] = dict(
                mean=summarize(rows, len(rows)),
                quality=summarize(quality_rows, len(rows)),
                rows=rows,
            )
            write_json(
                args.output.with_suffix(".progress.json"), dict(conditions=conditions)
            )
            print(
                seed,
                condition,
                {
                    m: r["metrics"]["macro_f1"]
                    for m, r in conditions[f"{seed}:{condition}"].items()
                    if m in ("mean", "quality")
                },
                flush=True,
            )
    write_json(
        args.output,
        dict(
            conditions=conditions,
            seconds=perf_counter() - began,
            cohort=[r["path"] for r in selected],
            seeds=list(SEEDS),
            selection="12 per source; sha256(path) ordering; no score-based selection",
            scope="post-crop 36-video validation pilot; 24 REAL / 12 FAKE; not full 602-video robustness or detector robustness",
            quality_scope="sharpness recomputed at 380 in clean AND corrupt conditions; bbox retention fixed",
            training_started=False,
            official_test_accessed=False,
        ),
    )


if __name__ == "__main__":
    main()
