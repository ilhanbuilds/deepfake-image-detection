"""Isolated, atomic 12-frame crop cache for the locked SBI train/val cohorts.

Training reads this cache only; missing crops never trigger implicit preprocessing.
The explicit cache CLI prepares both splits and resumes at completed videos.
"""

import argparse
from contextlib import contextmanager
import hashlib
from importlib.metadata import version
import os
from pathlib import Path
import shutil
from time import perf_counter
import zlib

import numpy as np
import torch

from src.train import atomic_save
from src.video_forensic import (
    SBIFaces,
    accepted_roster,
    read_json,
    summarize,
    validate_crops,
    write_json,
)
from src.video_frame_study import CHECKPOINT_SHA, STUDY, nested_indices, validate_cache
from src.video_manifest import digest_json
from src.video_pipeline import file_sha256

WORK = Path("outputs/video_sbi12_v1")
BASELINE = STUDY / "full.json"


@contextmanager
def exclusive_lock(path):
    """OS lock releases on crashes; the persistent lock file is not a stale lock."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as handle:
        handle.seek(0, os.SEEK_END)
        if handle.tell() == 0:
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            raise RuntimeError(f"Another process owns {path}") from error
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == "nt":
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def load_baseline(manifest, records):
    study = read_json(BASELINE)
    baseline = study["methods"]["12"]
    provenance = study["provenance"]
    expected = {r["path"]: r["label"] for r in records}
    if (
        not study["baseline_verified"]
        or provenance["checkpoint_sha256"] != CHECKPOINT_SHA
        or provenance["manifest_fingerprint"] != manifest["fingerprint"]
        or provenance["roster_sha256"] != digest_json(records)
        or provenance["precision"] != "fp32"
        or provenance["threshold"] != 0.5
        or provenance["test_accessed"]
        or provenance["training_steps"] != 0
        or baseline["coverage"] != 1
        or len(baseline["rows"]) != len(expected)
        or {r["path"]: r["target"] for r in baseline["rows"]} != expected
        or any(len(r["selected_indices"]) != 12 for r in baseline["rows"])
    ):
        raise ValueError("12-frame zero-shot baseline provenance/cohort mismatch.")
    recomputed = summarize(baseline["rows"], len(records))
    if any(recomputed[k] != baseline[k] for k in ("metrics", "sources")):
        raise ValueError("Baseline summary differs from its recorded predictions.")
    return study


def comparison(evaluated, baseline):
    if (
        evaluated["coverage"] != 1
        or len(evaluated["rows"]) != len(baseline["rows"])
        or {r["path"]: r["target"] for r in evaluated["rows"]}
        != {r["path"]: r["target"] for r in baseline["rows"]}
    ):
        raise ValueError("Baseline comparison requires the exact complete cohort.")
    metrics = ("accuracy", "macro_f1", "roc_auc", "balanced_accuracy")
    delta = {k: evaluated["metrics"][k] - baseline["metrics"][k] for k in metrics}
    delta.update(
        {
            f"{name.lower()}_recall": evaluated["metrics"]["per_class"][name]["recall"]
            - baseline["metrics"]["per_class"][name]["recall"]
            for name in ("REAL", "FAKE")
        }
    )
    return dict(
        baseline_metrics=baseline["metrics"],
        delta=delta,
        sources={
            source: {
                k: evaluated["sources"][source][k] - values[k]
                for k in ("recall", "errors", "roc_auc_vs_celeb_synthesis")
                if k in values
            }
            for source, values in baseline["sources"].items()
        },
        interpretation="Paired reused validation; not an independent test result.",
    )


def pixel_hash(pixels):
    return hashlib.sha256(pixels.numpy().tobytes()).hexdigest()


def save_crops(cached, path):
    # Lossless compression matters on the 4 GB laptop's nearly full system disk.
    packed = dict(
        cached, crops=zlib.compress(cached["crops"].numpy().tobytes(), level=1)
    )
    atomic_save(packed, path)


def read_crops(path):
    cached = torch.load(path, map_location="cpu", weights_only=True)
    count = sum(len(frame["faces"]) for frame in cached["report"]["frames"])
    expected = count * 3 * 380 * 380
    decoder = zlib.decompressobj()
    raw = decoder.decompress(cached["crops"], expected + 1)
    if len(raw) != expected or not decoder.eof or decoder.unused_data:
        raise ValueError("Compressed crop length/integrity mismatch.")
    cached["crops"] = torch.from_numpy(
        np.frombuffer(raw, dtype=np.uint8).copy()
    ).reshape(count, 3, 380, 380)
    return cached


def validate_payload(cached, record, signature, historical):
    report = cached["report"]
    pixels = cached["crops"]
    metadata = {k: v for k, v in cached.items() if k not in ("crops", "integrity")}
    if (
        cached["record"] != record
        or cached["signature"] != signature
        or cached["source_sha256"] != historical["source_sha256"]
        or cached["integrity"] != digest_json(metadata)
        or cached["pixels_sha256"] != pixel_hash(pixels)
        or report["path"] != record["path"]
        or report["selected_indices"]
        != nested_indices(
            historical["report"]["frame_count"],
            historical["report"]["selected_indices"],
        )[12]
    ):
        raise ValueError("12-frame cache record/source/pixels/protocol mismatch.")
    validate_crops(pixels, report)
    indices = [f["index"] for f in report["frames"]] + [
        f["index"] for f in report["failures"]
    ]
    if sorted(indices) != report["selected_indices"] or any(
        not failure.get("reason") for failure in report["failures"]
    ):
        raise ValueError("Missing/duplicate frame or unreported failure.")
    if report["status"] != "accepted":
        raise ValueError("Accepted cohort lost a video; cache preparation stopped.")


class TwelveFrameCache:
    def __init__(self, manifest, audit, train, val):
        self.study = load_baseline(manifest, val)
        self.baseline = self.study["methods"]["12"]
        self.original_signature = self.study["provenance"]["preprocessing"]
        self.signature = dict(
            self.original_signature,
            schema="sbi_nested_twelve_crop_v1",
            selection=self.study["provenance"]["sampler"],
            frames=12,
            storage="lossless_zlib_level1_RGB_uint8",
            manifest_fingerprint=manifest["fingerprint"],
            roster_hashes={"train": digest_json(train), "val": digest_json(val)},
            historical_cache_fingerprint=audit["cache_fingerprint"],
            implementation_sha256=file_sha256(Path(__file__)),
            sampler_sha256=file_sha256(Path("src/video_frame_study.py")),
            runtime_versions={
                name: version(name)
                for name in (
                    "torch",
                    "numpy",
                    "opencv-python-headless",
                    "retinaface-pytorch",
                    "albumentations",
                )
            },
        )
        self.directory = WORK / "crops" / digest_json(self.signature)
        self.records = {r["path"]: r for r in train + val}
        self.validation_paths = {r["path"] for r in val}
        self.extractor = None
        self.prepared = []

    def historical(self, record, audit):
        if self.records.get(record["path"]) != record:
            raise ValueError("Record outside locked train/val cohort.")
        old = torch.load(
            Path(audit["cache_directory"]) / (digest_json(record) + ".pt"),
            map_location="cpu",
            weights_only=True,
        )
        if old["record"] != record or old["signature"] != audit["cache_signature"]:
            raise ValueError("Historical cache changed.")
        return old

    def path(self, record):
        return self.directory / (digest_json(record) + ".pt")

    def read(self, root, record, audit):
        old = self.historical(record, audit)
        path = self.path(record)
        if not path.is_file():
            raise FileNotFoundError(
                f"Prepare the 12-frame cache first: {record['path']}"
            )
        cached = read_crops(path)
        validate_payload(cached, record, self.signature, old)
        return cached["crops"], cached["report"]

    def verify_source(self, root, record, old):
        source = (root / record["path"]).resolve()
        if not source.is_relative_to(root.resolve()):
            raise ValueError("Video escaped dataset root.")
        if file_sha256(source) != old["source_sha256"]:
            raise ValueError(f"Source bytes changed: {record['path']}")
        return source

    def verify_study_pixels(self, record, pixels, report):
        if record["path"] not in self.validation_paths:
            return
        cached = read_json(STUDY / "frames" / (digest_json(record) + ".json"))
        validate_cache(cached, record, self.study["provenance"])
        expected = [
            f for f in cached["frames"] if f["index"] in report["selected_indices"]
        ]
        if [f["index"] for f in expected if f["status"] == "accepted"] != [
            f["index"] for f in report["frames"]
        ]:
            raise ValueError("Frame-study face coverage changed.")
        actual = {f["index"]: f for f in report["frames"]}
        for frame in expected:
            if frame["status"] != "accepted":
                continue
            faces = actual[frame["index"]]["faces"]
            if len(faces) != len(frame["faces"]):
                raise ValueError("Frame-study face count changed.")
            for a, b in zip(faces, frame["faces"], strict=True):
                if (
                    a["bbox"] != b["bbox"]
                    or a["crop_size"] != b["crop_size"]
                    or pixel_hash(pixels[a["crop_index"]]) != b["crop_sha256"]
                ):
                    raise ValueError("Frame-study bbox/crop pixels changed.")

    def prepare(self, root, record, audit):
        began = perf_counter()
        old = self.historical(record, audit)
        source = self.verify_source(root, record, old)
        if self.path(record).exists():
            pixels, report = self.read(root, record, audit)
            reused = True
        else:
            if self.extractor is None:
                self.extractor = SBIFaces("cuda")
                if self.extractor.signature != self.original_signature:
                    raise ValueError("Official preprocessing changed.")
            self.directory.mkdir(parents=True, exist_ok=True)
            if shutil.disk_usage(self.directory).free < 128 * 1024**2:
                raise OSError(
                    "Insufficient free disk space for the next atomic crop file."
                )
            pixels, report = extract(self.extractor, source, record, old["report"])
            self.verify_study_pixels(record, pixels, report)
            cached = dict(
                record=record,
                signature=self.signature,
                source_sha256=old["source_sha256"],
                crops=pixels,
                report=report,
                pixels_sha256=pixel_hash(pixels),
            )
            cached["integrity"] = digest_json(
                {k: v for k, v in cached.items() if k != "crops"}
            )
            validate_payload(cached, record, self.signature, old)
            save_crops(cached, self.path(record))
            reused = False
        self.verify_study_pixels(record, pixels, report)
        result = dict(
            path=record["path"],
            reused=reused,
            seconds=perf_counter() - began,
            bytes=self.path(record).stat().st_size,
            valid_frames=len(report["frames"]),
            failures=report["failures"],
            unavailable_frames=12 - len(report["selected_indices"]),
        )
        self.prepared.append(result)
        return result

    def release_detector(self):
        self.extractor = None
        torch.cuda.empty_cache()

    def preflight(self, root, records, audit):
        """Read/hash only: never decode or prepare missing cache during training."""
        for index, record in enumerate(records, 1):
            old = self.historical(record, audit)
            self.verify_source(root, record, old)
            pixels, report = self.read(root, record, audit)
            self.verify_study_pixels(record, pixels, report)
            if index % 100 == 0:
                print(f"cache verified={index}/{len(records)}", flush=True)


@torch.inference_mode()
def extract(extractor, source, record, historical):
    cv = extractor.cv
    indices = nested_indices(historical["frame_count"], historical["selected_indices"])[
        12
    ]
    report = dict(path=record["path"], selected_indices=indices, frames=[], failures=[])
    crops = []
    cap = cv.VideoCapture(str(source))
    try:
        if (
            not cap.isOpened()
            or int(cap.get(cv.CAP_PROP_FRAME_COUNT)) != historical["frame_count"]
        ):
            raise ValueError("Video unreadable or frame count changed.")
        for index in range(indices[-1] + 1):
            if not cap.grab():
                report["failures"].extend(
                    dict(index=i, reason="decode_failed") for i in indices if i >= index
                )
                break
            if index not in indices:
                continue
            ok, frame = cap.retrieve()
            if not ok or frame is None:
                report["failures"].append(dict(index=index, reason="decode_failed"))
                continue
            rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            faces = [
                f for f in extractor.detector.predict_jsons(rgb) if len(f["bbox"]) == 4
            ]
            areas = [
                (f["bbox"][2] - f["bbox"][0]) * (f["bbox"][3] - f["bbox"][1])
                for f in faces
            ]
            valid = []
            for face, area in zip(faces, areas, strict=True):
                if area <= 0 or area < max(areas) / 2:
                    continue
                crop = extractor.crop_face(
                    rgb,
                    None,
                    np.asarray(face["bbox"]).reshape(2, 2),
                    False,
                    crop_by_bbox=True,
                    only_img=True,
                    phase="test",
                )
                if not crop.size:
                    continue
                pixels = cv.resize(crop, (380, 380)).transpose(2, 0, 1).copy()
                gray = cv.cvtColor(crop, cv.COLOR_RGB2GRAY)
                crops.append(torch.from_numpy(pixels))
                valid.append(
                    dict(
                        crop_index=len(crops) - 1,
                        bbox=face["bbox"],
                        sharpness=float(np.log1p(cv.Laplacian(gray, cv.CV_64F).var())),
                        crop_size=list(crop.shape[:2]),
                    )
                )
            if valid:
                report["frames"].append(dict(index=index, faces=valid))
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
    return pixels, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-cache", action="store_true", required=True)
    parser.add_argument("--resume", choices=("auto",), required=True)
    parser.parse_args()
    torch.set_num_threads(4)
    manifest, audit, train = accepted_roster("train")
    _, _, val = accepted_roster("val")
    cache = TwelveFrameCache(manifest, audit, train, val)
    root = Path(manifest["dataset_root"])
    with exclusive_lock(WORK / "cache.lock"):
        began = perf_counter()
        try:
            for split, records in (("train", train), ("val", val)):
                for index, record in enumerate(records, 1):
                    result = cache.prepare(root, record, audit)
                    print(
                        f"{split} {index}/{len(records)} reused={result['reused']} seconds={result['seconds']:.2f} {record['path']}",
                        flush=True,
                    )
                    write_json(
                        cache.directory / "progress.json",
                        dict(split=split, index=index, last=result),
                    )
            write_json(
                cache.directory / "complete.json",
                dict(
                    signature=cache.signature,
                    train=len(train),
                    validation=len(val),
                    seconds=perf_counter() - began,
                    records=cache.prepared,
                    official_test_accessed=False,
                ),
            )
        except BaseException as error:
            write_json(
                cache.directory / "interrupted.json",
                dict(
                    reason=type(error).__name__,
                    message=str(error),
                    completed_this_invocation=len(cache.prepared),
                    recovery="Rerun the identical command; verified completed videos are reused. Corrupt caches fail closed.",
                ),
            )
            raise
        finally:
            cache.release_detector()


if __name__ == "__main__":
    main()
