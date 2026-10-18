"""Isolated SBI inference, with explicit provenance and accepted train/val rosters.

No test-set decoder, image-app imports or changes to the historical crop cache.
The upstream research-only model and preprocessing live outside source control.
"""

import argparse
import importlib.util
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from torch import nn

from src.video_evaluate import calculate_video_metrics
from src.video_manifest import digest_json, load_manifest
from src.video_pipeline import file_sha256, opencv
from src.video_spatial_evaluate import spatial_source_metrics

ROOT = Path("outputs/video_research_v1")
OFFICIAL_HASHES = {
    "e477fc1789f8c1d2147cffe86a47eb37a24adedae10b106b7d47c1eeb73777e2": "FFraw",
    "2b05c2ba36ccb9e9f4f9e1aae9acd443ae2e6400ce725f56104fdb175d3c3267": "FFc23",
}


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def accepted_roster(split):
    if split not in ("train", "val"):
        raise ValueError("Only accepted train/val; official test is locked.")
    manifest = load_manifest(
        Path("outputs/video_identity_v1/manifest.json"), verify_inventory=False
    )
    audit = read_json(ROOT / "roster_audit.json")
    if manifest["fingerprint"] != audit["manifest_fingerprint"]:
        raise ValueError("Manifest/audit mismatch.")
    records = audit["splits"][split]["accepted_records"]
    expected = {r["path"]: r for r in manifest["splits"][split]}
    if any(expected.get(r["path"]) != r for r in records):
        raise ValueError("Accepted roster escaped its manifest split.")
    counts = {label: sum(r["label"] == label for r in records) for label in (0, 1)}
    if counts != ({0: 417, 1: 1806} if split == "train" else {0: 106, 1: 496}):
        raise ValueError("Historical accepted roster changed.")
    return manifest, audit, records


class SBIClassifier(nn.Module):
    def __init__(self, checkpoint):
        super().__init__()
        from efficientnet_pytorch import EfficientNet

        # No extra ImageNet download: strict official SBI state replaces ALL weights.
        self.net = EfficientNet.from_name("efficientnet-b4", num_classes=2)
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        checksum = file_sha256(checkpoint)
        self.adaptation_config = None
        if checksum in OFFICIAL_HASHES:
            self.checkpoint_kind = OFFICIAL_HASHES[checksum]
            state = saved["model"]
        elif saved.get("pipeline") == "sbi_tail_adaptation_v1":
            config = saved.get("config", {})
            if (
                config.get("class_names") != ["REAL", "FAKE"]
                or config.get("preprocessing", {}).get("resolution") != 380
            ):
                raise ValueError("Adapted checkpoint class/preprocessing mismatch.")
            self.checkpoint_kind = "local_sbi_adaptation"
            self.adaptation_config = config
            state = saved["model_state_dict"]
        else:
            raise ValueError("Unverified SBI checkpoint provenance.")
        self.load_state_dict(state, strict=True)

    def forward(self, images):
        return self.net(images)


def official_crop_function():
    path = ROOT / "upstream/src_inference_preprocess.py"
    if (
        file_sha256(path)
        != "cd9d03e4e85fa39fb83e8bf6fe0756ed3a1a92c32fe252918225490cb018341e"
    ):
        raise ValueError(
            "Official crop source changed; do not silently change preprocessing."
        )
    spec = importlib.util.spec_from_file_location("sbi_upstream_preprocess", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.crop_face


class SBIFaces:
    """Official RetinaFace, bbox margins and resize; six recorded sample indices.

    Frames with multiple faces retain >= half-largest area, as upstream inference.
    Decode errors and absent detections are retained; no neutral-score imputation.
    """

    def __init__(self, device="cuda"):
        from retinaface.predict_single import Model

        self.cv = opencv()
        self.crop_face = official_crop_function()
        self.detector = Model(max_size=2048, device=device)
        if (
            file_sha256(ROOT / "retinaface.pth")
            != "b9fdd58c474d809b67166130b42d5913f00b48f912a3491565bc56b8a8a796f2"
        ):
            raise ValueError("RetinaFace checkpoint checksum mismatch.")
        self.detector.load_state_dict(
            torch.load(ROOT / "retinaface.pth", map_location="cpu", weights_only=True)
        )
        self.detector.eval()
        self.device = device
        self.signature = dict(
            schema="sbi_official_crop_six_frames_v1",
            detector_sha256=file_sha256(ROOT / "retinaface.pth"),
            crop_code_sha256=file_sha256(ROOT / "upstream/src_inference_preprocess.py"),
            resolution=380,
            detector_max_size=2048,
            detector_device=device,
            pixel_contract="RGB uint8 -> float32 / 255; no ImageNet normalization",
            selection="historical six selected_indices; max face score per frame",
            minimum_frames=3,
        )

    def read(self, root, record, audit):
        baseline = torch.load(
            Path(audit["cache_directory"]) / (digest_json(record) + ".pt"),
            map_location="cpu",
            weights_only=True,
        )
        if (
            baseline.get("record") != record
            or baseline.get("signature") != audit["cache_signature"]
        ):
            raise ValueError("Historical cache record/signature mismatch.")
        source = (root / record["path"]).resolve()
        if not source.is_relative_to(root.resolve()):
            raise ValueError("Video escaped dataset root.")
        if file_sha256(source) != baseline["source_sha256"]:
            raise ValueError("Source video bytes changed.")
        key = digest_json(dict(record=record, signature=self.signature))
        cache_path = ROOT / "sbi_crops" / (key + ".pt")
        if cache_path.exists():
            cached = torch.load(cache_path, map_location="cpu", weights_only=True)
            if (
                cached["source_sha256"] != baseline["source_sha256"]
                or cached["signature"] != self.signature
            ):
                raise ValueError("SBI crop cache mismatch.")
            validate_crops(cached["crops"], cached["report"])
            return cached["crops"], cached["report"]
        indices = baseline["report"]["selected_indices"]
        report = dict(
            path=record["path"], selected_indices=indices, frames=[], failures=[]
        )
        crops = []
        cap = self.cv.VideoCapture(str(source))
        try:
            if not cap.isOpened():
                raise RuntimeError("video_open_failed")
            for index in range(indices[-1] + 1):
                if not cap.grab():
                    report["failures"].extend(
                        dict(index=i, reason="decode_failed")
                        for i in indices
                        if i >= index
                    )
                    break
                if index not in indices:
                    continue
                ok, frame = cap.retrieve()
                if not ok or frame is None:
                    report["failures"].append(dict(index=index, reason="decode_failed"))
                    continue
                rgb = self.cv.cvtColor(frame, self.cv.COLOR_BGR2RGB)
                faces = [
                    f for f in self.detector.predict_jsons(rgb) if len(f["bbox"]) == 4
                ]
                areas = [
                    (f["bbox"][2] - f["bbox"][0]) * (f["bbox"][3] - f["bbox"][1])
                    for f in faces
                ]
                valid_faces = []
                for face, area in zip(faces, areas, strict=True):
                    if area <= 0 or area < max(areas) / 2:
                        continue
                    box = np.asarray(face["bbox"]).reshape(2, 2)
                    crop = self.crop_face(
                        rgb,
                        None,
                        box,
                        False,
                        crop_by_bbox=True,
                        only_img=True,
                        phase="test",
                    )
                    if not crop.size:
                        continue
                    pixels = self.cv.resize(crop, (380, 380)).transpose(2, 0, 1).copy()
                    gray = self.cv.cvtColor(crop, self.cv.COLOR_RGB2GRAY)
                    quality = float(
                        np.log1p(self.cv.Laplacian(gray, self.cv.CV_64F).var())
                    )
                    crops.append(torch.from_numpy(pixels))
                    valid_faces.append(
                        dict(
                            crop_index=len(crops) - 1,
                            bbox=face["bbox"],
                            sharpness=quality,
                            crop_size=list(crop.shape[:2]),
                        )
                    )
                if valid_faces:
                    report["frames"].append(dict(index=index, faces=valid_faces))
                else:
                    report["failures"].append(dict(index=index, reason="no_valid_face"))
        finally:
            cap.release()
        report["status"] = "accepted" if len(report["frames"]) >= 3 else "rejected"
        pixels = (
            torch.stack(crops)
            if crops
            else torch.empty((0, 3, 380, 380), dtype=torch.uint8)
        )
        from src.train import atomic_save

        validate_crops(pixels, report)
        atomic_save(
            dict(
                crops=pixels,
                report=report,
                source_sha256=baseline["source_sha256"],
                signature=self.signature,
            ),
            cache_path,
        )
        return pixels, report


def validate_crops(pixels, report):
    if (
        pixels.dtype != torch.uint8
        or pixels.ndim != 4
        or pixels.shape[1:] != (3, 380, 380)
    ):
        raise ValueError("SBI cache must contain RGB uint8 [N,3,380,380].")
    frames = report["frames"]
    selected = report["selected_indices"]
    indices = [f["index"] for f in frames]
    crop_indices = [face["crop_index"] for frame in frames for face in frame["faces"]]
    if (
        indices != sorted(set(indices))
        or not set(indices).issubset(selected)
        or crop_indices != list(range(len(pixels)))
        or any(not f["faces"] for f in frames)
        or report["status"] != ("accepted" if len(frames) >= 3 else "rejected")
    ):
        raise ValueError("SBI cache frame/face/status integrity failure.")


@torch.inference_mode()
def score_faces(model, crops, device):
    if (
        crops.dtype != torch.uint8
        or crops.ndim != 4
        or crops.shape[1:] != (3, 380, 380)
    ):
        raise ValueError("Expected unnormalized uint8 RGB [N,3,380,380].")
    scores = []
    for crop in crops:
        logits = model(crop[None].to(device=device, dtype=torch.float32) / 255)
        if logits.shape != (1, 2) or not torch.isfinite(logits).all():
            raise ValueError("SBI logits invalid.")
        scores.append(float(logits.softmax(1)[0, 1]))
    return scores


def frame_scores(face_scores, report):
    return [
        max(face_scores[f["crop_index"]] for f in frame["faces"])
        for frame in report["frames"]
    ]


def summarize(rows, attempted):
    valid = [r for r in rows if r["status"] == "accepted"]
    metrics = (
        calculate_video_metrics(
            [r["target"] for r in valid],
            [r["prediction"] for r in valid],
            [r["fake_probability"] for r in valid],
        )
        if valid
        else None
    )
    return dict(
        metrics=metrics,
        sources=spatial_source_metrics(valid) if valid else None,
        attempted=attempted,
        evaluated=len(valid),
        coverage=len(valid) / attempted if attempted else 0,
        rejected=[r for r in rows if r["status"] != "accepted"],
        threshold=0.5,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Isolated SBI zero-shot; train/val only, never optimizer steps."
    )
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "FFraw.tar")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Existing result is protected.")
    manifest, audit, records = accepted_roster("train" if args.smoke else "val")
    if args.smoke:
        records = [
            next(r for r in records if r["path"].startswith(s + "/"))
            for s in ("Celeb-real", "YouTube-real", "Celeb-synthesis")
        ]
    torch.set_num_threads(4)
    model = SBIClassifier(args.checkpoint).to(args.device).eval()
    extractor = SBIFaces(args.device)
    if model.adaptation_config is not None and (
        model.adaptation_config["manifest_fingerprint"] != manifest["fingerprint"]
        or model.adaptation_config["preprocessing"] != extractor.signature
    ):
        raise ValueError("Adapted checkpoint manifest/preprocessing mismatch.")
    if args.device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    rows = []
    began = perf_counter()
    provenance = dict(
        checkpoint=str(args.checkpoint),
        checkpoint_sha256=file_sha256(args.checkpoint),
        manifest_fingerprint=manifest["fingerprint"],
        preprocessing=extractor.signature,
        split="train_smoke" if args.smoke else "accepted_validation",
        training_steps=0,
        checkpoint_kind=model.checkpoint_kind,
        checkpoint_train_data="FF++ original real videos and synthetic SBIs; per upstream"
        if model.adaptation_config is None
        else "FF++ SBI pretraining plus accepted local train video adaptation",
        test_accessed=False,
        precision="fp32",
    )
    for n, record in enumerate(records, 1):
        started = perf_counter()
        try:
            pixels, report = extractor.read(
                Path(manifest["dataset_root"]), record, audit
            )
            scores = score_faces(model, pixels, args.device)
        except BaseException as error:
            write_json(
                args.output.with_suffix(".interrupted.json"),
                dict(
                    provenance=provenance,
                    failed_path=record["path"],
                    completed=len(rows),
                    expected=len(records),
                    error_type=type(error).__name__,
                    message=str(error),
                    silent_skips=0,
                    recovery="Re-run with a new output path; verified SBI crop cache is reusable.",
                ),
            )
            raise
        per_frame = frame_scores(scores, report)
        row = dict(
            path=record["path"],
            target=record["label"],
            status=report["status"],
            frame_scores=per_frame,
            face_scores=scores,
            report=report,
            seconds=perf_counter() - started,
        )
        if report["status"] == "accepted":
            probability = float(np.mean(per_frame))
            row.update(fake_probability=probability, prediction=int(probability >= 0.5))
        rows.append(row)
        write_json(
            args.output.with_suffix(".progress.json"),
            dict(provenance=provenance, completed=n, expected=len(records), rows=rows),
        )
        if n % 10 == 0 or args.smoke:
            print(
                f"SBI {n}/{len(records)} {record['path']} {row.get('fake_probability')} elapsed={perf_counter() - began:.1f}s",
                flush=True,
            )
    result = dict(
        provenance=provenance,
        **summarize(rows, len(records)),
        rows=rows,
        seconds=perf_counter() - began,
        peak_allocated_mib=torch.cuda.max_memory_allocated() / 1024**2
        if args.device == "cuda"
        else None,
        peak_reserved_mib=torch.cuda.max_memory_reserved() / 1024**2
        if args.device == "cuda"
        else None,
    )
    write_json(args.output, result)
    print(
        json.dumps(
            {
                k: v
                for k, v in result.items()
                if k not in ("rows", "provenance", "rejected")
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
