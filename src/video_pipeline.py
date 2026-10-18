"""One decoder/detector/crop path shared by smoke, training, cache and inference."""

import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from src.video_manifest import digest_json


def opencv():
    try:
        import cv2
    except ImportError as error:
        raise RuntimeError(
            "Video için: .\\.venv\\Scripts\\python.exe -m pip install -r requirements-video.txt"
        ) from error
    return cv2


@dataclass(frozen=True)
class VideoSettings:
    frames: int = 6
    image_size: int = 160
    min_valid_frames: int = 3
    margin: float = 0.2
    detection_max_side: int = 640
    scale_factor: float = 1.1
    min_neighbors: int = 5
    min_face_size: int = 30
    min_track_iou: float = 0.1

    def __post_init__(self):
        if self.frames != 6 or self.image_size not in (160, 192):
            raise ValueError("Video V1: 6 frame ve 160/192 çözünürlük.")
        if not 1 <= self.min_valid_frames <= self.frames or not 0 <= self.margin <= 1:
            raise ValueError("Minimum yüz sayısı/margin geçersiz.")
        if (
            self.detection_max_side < self.min_face_size
            or self.min_face_size < 1
            or self.scale_factor <= 1
            or self.min_neighbors < 1
            or not 0 < self.min_track_iou <= 1
        ):
            raise ValueError("Yüz detector/track ayarları geçersiz.")


def sample_indices(frame_count: int, count: int = 6) -> list[int]:
    if frame_count < count or count < 2:
        raise ValueError(
            "Altı farklı frame için video çok kısa veya frame sayısı geçersiz."
        )
    return np.linspace(0, frame_count - 1, count).round().astype(int).tolist()


def box_iou(a: tuple, b: tuple) -> float:
    left, top = max(a[0], b[0]), max(a[1], b[1])
    right, bottom = min(a[0] + a[2], b[0] + b[2]), min(a[1] + a[3], b[1] + b[3])
    intersection = max(0, right - left) * max(0, bottom - top)
    return intersection / max(1, a[2] * a[3] + b[2] * b[3] - intersection)


class FacePipeline:
    """CPU Haar cascade, largest initial face then IoU track; no full-frame fallback."""

    def __init__(self, settings: VideoSettings = VideoSettings()):
        self.settings = settings
        self.cv = opencv()
        cascade_path = (
            Path(self.cv.data.haarcascades) / "haarcascade_frontalface_default.xml"
        )
        self.detector = self.cv.CascadeClassifier(str(cascade_path))
        if self.detector.empty():
            raise RuntimeError("OpenCV Haar yüz detector yüklenemedi.")
        self.signature = dict(
            settings=asdict(settings),
            version=1,
            decoder="opencv sequential BGR to RGB",
            detector="haarcascade_frontalface_default",
            opencv=self.cv.__version__,
            detector_sha256=hashlib.sha256(cascade_path.read_bytes()).hexdigest(),
            normalization="ImageNet V1",
            interpolation="PIL bilinear antialias",
            selection="largest initial face, subsequent IoU >= threshold; no fallback",
        )
        self.transform = transforms.Compose(
            [
                transforms.Resize(
                    (settings.image_size, settings.image_size), antialias=True
                ),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )

    def crop(
        self, bgr: np.ndarray, previous: tuple | None
    ) -> tuple[Image.Image | None, tuple | None, str | None]:
        cv, settings = self.cv, self.settings
        height, width = bgr.shape[:2]
        scale = min(1.0, settings.detection_max_side / max(height, width))
        gray = cv.cvtColor(bgr, cv.COLOR_BGR2GRAY)
        if scale < 1:
            gray = cv.resize(
                gray, None, fx=scale, fy=scale, interpolation=cv.INTER_AREA
            )
        faces = self.detector.detectMultiScale(
            gray,
            scaleFactor=settings.scale_factor,
            minNeighbors=settings.min_neighbors,
            minSize=(settings.min_face_size, settings.min_face_size),
        )
        boxes = [tuple(int(round(int(v) / scale)) for v in box) for box in faces]
        if not boxes:
            return None, previous, "face_not_detected"
        if previous is None:
            box = max(boxes, key=lambda b: (b[2] * b[3], -b[0], -b[1]))
        else:
            box = max(boxes, key=lambda b: (box_iou(b, previous), b[2] * b[3], -b[0]))
            if box_iou(box, previous) < settings.min_track_iou:
                return None, previous, "face_track_lost"
        x, y, w, h = box
        side = max(w, h) * (1 + 2 * settings.margin)
        x0, y0 = max(0, int(x + w / 2 - side / 2)), max(0, int(y + h / 2 - side / 2))
        x1, y1 = (
            min(width, int(x + w / 2 + side / 2)),
            min(height, int(y + h / 2 + side / 2)),
        )
        if x1 <= x0 or y1 <= y0:
            return None, previous, "invalid_face_box"
        rgb = cv.cvtColor(bgr[y0:y1, x0:x1], cv.COLOR_BGR2RGB)
        return Image.fromarray(rgb), box, None

    def read(self, path: Path) -> tuple[torch.Tensor | None, dict, list[Image.Image]]:
        cap = self.cv.VideoCapture(str(path))
        report = dict(
            path=str(path),
            requested_frames=self.settings.frames,
            valid_frames=0,
            failed_frames=[],
            selected_indices=[],
            valid_indices=[],
            boxes=[],
            status="rejected",
        )
        crops = []
        try:
            if not cap.isOpened():
                report["error"] = "video_open_failed"
                return None, report, crops
            raw_count = cap.get(self.cv.CAP_PROP_FRAME_COUNT)
            if not np.isfinite(raw_count) or raw_count < self.settings.frames:
                report["error"] = "invalid_or_short_frame_count"
                return None, report, crops
            indices = sample_indices(int(raw_count), self.settings.frames)
            report["selected_indices"] = indices
            fps = cap.get(self.cv.CAP_PROP_FPS)
            report["fps"] = float(fps) if np.isfinite(fps) and fps > 0 else None
            report["frame_count"] = int(raw_count)
            report["duration_seconds"] = int(raw_count) / fps if report["fps"] else None
            previous = None
            # Sequential decoding avoids backend/keyframe seek silently returning wrong frames.
            next_sample = 0
            for index in range(indices[-1] + 1):
                if not cap.grab():
                    report["failed_frames"].extend(
                        dict(index=i, reason="decode_failed")
                        for i in indices[next_sample:]
                    )
                    break
                if index != indices[next_sample]:
                    continue
                ok, bgr = cap.retrieve()
                if not ok or bgr is None:
                    report["failed_frames"].append(
                        dict(index=index, reason="decode_failed")
                    )
                else:
                    crop, previous, reason = self.crop(bgr, previous)
                    if crop is None:
                        report["failed_frames"].append(dict(index=index, reason=reason))
                    else:
                        crops.append(crop)
                        report["valid_indices"].append(index)
                        report["boxes"].append(list(previous))
                next_sample += 1
                if next_sample == len(indices):
                    break
            report["valid_frames"] = len(crops)
            if len(crops) < self.settings.min_valid_frames:
                report["error"] = "insufficient_valid_faces"
                return None, report, crops
            report["status"] = "accepted"
            return torch.stack([self.transform(crop) for crop in crops]), report, crops
        finally:
            cap.release()


class VideoDataset(Dataset):
    """One video item and label; decode failures retained for explicit coverage reports."""

    def __init__(self, root: Path, records: list[dict], settings: VideoSettings):
        self.root, self.records, self.settings = root, records, settings
        self.pipeline = None

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        if self.pipeline is None:
            self.pipeline = FacePipeline(self.settings)
        record = self.records[index]
        frames, report, _ = self.pipeline.read(self.root / record["path"])
        report.update(relative_path=record["path"], label=record["label"])
        return dict(record=record, values=frames, report=report)


class FeatureDataset(Dataset):
    def __init__(self, records: list[dict], cache, root: Path):
        self.records, self.cache, self.root = records, cache, root

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        return self.cache.read(record, self.root)


def video_collate(items: list[dict]) -> dict:
    valid = [item for item in items if item["values"] is not None]
    return dict(
        items=valid,
        reports=[item["report"] for item in items],
        labels=torch.tensor(
            [item["record"]["label"] for item in valid], dtype=torch.float32
        ),
    )


def make_video_loader(
    dataset, batch_size: int = 2, shuffle: bool = False, seed: int = 42
):
    if batch_size not in (1, 2):
        raise ValueError("GTX 1650 başlangıcı için batch size 1 veya 2 kullanın.")
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        collate_fn=video_collate,
        generator=torch.Generator().manual_seed(seed),
    )


def feature_batch(model, batch: dict, device: torch.device, cached: bool, amp: bool):
    sequences = []
    for item in batch["items"]:
        values = item["values"].to(device)
        if not cached:
            with torch.amp.autocast(device.type, enabled=amp):
                values = model.extract_features(values)
        sequences.append(values.float())
    lengths = torch.tensor([len(values) for values in sequences], dtype=torch.long)
    return pad_sequence(sequences, batch_first=True), lengths


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class FeatureCache:
    """Content-addressed frozen features; validates pipeline AND actual backbone weights."""

    def __init__(self, directory: Path, pipeline_signature: dict, backbone_hash: str):
        self.signature = dict(
            pipeline=pipeline_signature,
            backbone_sha256=backbone_hash,
            schema=1,
            feature_dim=1280,
            feature_dtype="float32",
            extraction="fp32",
        )
        self.directory = directory / digest_json(self.signature)
        self.verified_sources = {}

    def path(self, record: dict) -> Path:
        return self.directory / (digest_json(record) + ".pt")

    def write(
        self, record: dict, root: Path, features: torch.Tensor | None, report: dict
    ) -> None:
        from src.train import atomic_save

        destination = self.path(record)
        if destination.exists():
            raise FileExistsError(f"Feature cache korunuyor: {destination}")
        atomic_save(
            dict(
                signature=self.signature,
                record=record,
                source_sha256=file_sha256(root / record["path"]),
                features=features.cpu() if features is not None else None,
                report=report,
            ),
            destination,
        )

    def read(self, record: dict, root: Path) -> dict:
        path = self.path(record)
        if not path.is_file():
            raise FileNotFoundError(
                f"Feature cache eksik; önce açık --cache-split komutu: {record['path']}"
            )
        saved = torch.load(path, map_location="cpu", weights_only=True)
        source = root / record["path"]
        stat = source.stat()
        identity = (str(source.resolve()), stat.st_size, stat.st_mtime_ns)
        # Hash each unchanged source once per process, not once per training epoch.
        if identity not in self.verified_sources:
            self.verified_sources[identity] = file_sha256(source)
        if (
            saved["signature"] != self.signature
            or saved["record"] != record
            or saved["source_sha256"] != self.verified_sources[identity]
        ):
            raise ValueError(f"Feature cache uyumsuz: {record['path']}")
        features, report = saved["features"], saved["report"]
        if report.get("status") not in {"accepted", "rejected"}:
            raise ValueError(f"Cache yüz durumu geçersiz: {path}")
        settings = self.signature["pipeline"].get("settings", {})
        minimum, maximum = (
            settings.get("min_valid_frames", 3),
            settings.get("frames", 6),
        )
        if features is not None and (
            not isinstance(features, torch.Tensor)
            or not minimum <= report["valid_frames"] <= maximum
            or features.shape != (report["valid_frames"], 1280)
            or not torch.isfinite(features).all()
            or features.dtype != torch.float32
        ):
            raise ValueError(f"Bozuk feature cache: {path}")
        if (features is None) != (report["status"] != "accepted"):
            raise ValueError(f"Cache yüz raporu uyumsuz: {path}")
        if features is None and not 0 <= report["valid_frames"] < minimum:
            raise ValueError(f"Cache ret kare sayısı uyumsuz: {path}")
        if settings and features is not None:
            indices = report.get("valid_indices", [])
            selected = report.get("selected_indices", [])
            failed = report.get("failed_frames", [])
            if (
                len(indices) != len(features)
                or indices != sorted(set(indices))
                or len(selected) != maximum
                or selected != sorted(set(selected))
                or not set(indices).issubset(selected)
                or set(indices) & {row["index"] for row in failed}
                or set(indices) | {row["index"] for row in failed} != set(selected)
                or len(indices) + len(failed) != maximum
            ):
                raise ValueError(f"Cache frame sırası/raporu uyumsuz: {path}")
        return dict(record=record, values=features, report=report)
