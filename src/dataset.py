"""Deterministik, sınıf oranlarını koruyan görüntü veri hattı."""

import random
import hashlib
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

SEED = 42
IMAGE_SIZE = 224
CLASS_NAMES = ("REAL", "FAKE")
CLASS_TO_IDX = {"Real": 0, "Fake": 1}
EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
Sample = tuple[str, int]


def seed_everything(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def seed_worker(worker_id: int) -> None:
    seed = torch.initial_seed() % (2**32)
    random.seed(seed)
    np.random.seed(seed)


def discover_samples(root: str | Path) -> list[Sample]:
    root = Path(root)
    if not root.is_dir():
        raise ValueError(f"Dataset klasörü bulunamadı: {root}")
    unexpected = [
        p.name for p in root.iterdir() if p.is_dir() and p.name not in CLASS_TO_IDX
    ]
    if unexpected:
        raise ValueError(
            f"Beklenmeyen sınıf klasörleri: {unexpected}. Real ve Fake kullanın."
        )
    samples: list[Sample] = []
    for folder, label in CLASS_TO_IDX.items():
        class_dir = root / folder
        if not class_dir.is_dir():
            raise ValueError(f"Sınıf klasörü bulunamadı: {class_dir}")
        paths = sorted(
            p
            for p in class_dir.rglob("*")
            if p.suffix.lower() in EXTENSIONS and p.is_file()
        )
        if not paths:
            raise ValueError(f"Sınıf klasöründe görüntü yok: {class_dir}")
        samples.extend((p.relative_to(root).as_posix(), label) for p in paths)
    return samples


def split_samples(
    samples: list[Sample], seed: int = SEED, groups: dict[str, str] | None = None
) -> dict[str, list[Sample]]:
    if groups is not None:
        grouped: dict[str, list[Sample]] = defaultdict(list)
        for sample in samples:
            grouped[groups[sample[0]]].append(sample)
        if any(len({label for _, label in subset}) != 1 for subset in grouped.values()):
            raise ValueError("Aynı görüntü grubu farklı sınıf etiketleri içeriyor.")
        splits: dict[str, list[Sample]] = {"train": [], "val": [], "test": []}
        rng = random.Random(seed)
        for label in (0, 1):
            families = [subset for subset in grouped.values() if subset[0][1] == label]
            if len(families) < 3:
                raise ValueError("Her sınıfta en az üç bağımsız görüntü grubu gerekli.")
            rng.shuffle(families)
            families.sort(key=len, reverse=True)
            count = sum(map(len, families))
            train_count = math.floor(count * 0.7)
            remaining = count - train_count
            targets = dict(
                train=train_count, val=remaining // 2, test=remaining - remaining // 2
            )
            assigned = dict.fromkeys(targets, 0)
            for family in families:
                name = max(targets, key=lambda key: targets[key] - assigned[key])
                splits[name].extend(family)
                assigned[name] += len(family)
        if any({label for _, label in subset} != {0, 1} for subset in splits.values()):
            raise ValueError(
                "Grup ayrımı iki sınıfı da içermeli; daha fazla bağımsız grup gerekli."
            )
        return splits
    try:
        train, remaining = train_test_split(
            samples, test_size=0.30, random_state=seed, stratify=[s[1] for s in samples]
        )
        val, test = train_test_split(
            remaining,
            test_size=0.50,
            random_state=seed,
            stratify=[s[1] for s in remaining],
        )
    except ValueError as error:
        raise ValueError(
            "%70/%15/%15 sınıflı ayrım için yeterli görüntü yok."
        ) from error
    splits = {"train": train, "val": val, "test": test}
    if any({label for _, label in subset} != {0, 1} for subset in splits.values()):
        raise ValueError(
            "Her ayrımda hem Real hem Fake görüntü bulunmalı; daha fazla veri ekleyin."
        )
    return splits


def audit_dataset(root: str | Path, seed: int = SEED, progress: bool = False) -> dict:
    """Dosyaları değiştirmeden tam decode, SHA256 ve bilinen kaynak gruplarını denetler."""
    root = Path(root).resolve()
    samples = discover_samples(root)
    parents = {path: path for path, _ in samples}

    def find(path: str) -> str:
        while parents[path] != path:
            parents[path] = parents[parents[path]]
            path = parents[path]
        return path

    def union(first: str, second: str) -> None:
        parents[find(second)] = find(first)

    seen: dict[tuple, str] = {}
    records = []
    errors = []
    duplicates = []
    path_errors = []
    augmentation_count = 0
    for index, (relative, label) in enumerate(samples, 1):
        path = root / relative
        resolved = path.resolve()
        if not resolved.is_relative_to(root) or path.is_symlink():
            path_errors.append(relative)
            continue
        try:
            file_hash = hashlib.sha256(path.read_bytes()).hexdigest()
            with Image.open(path) as image:
                image.verify()
            image = read_image(path)
            image.load()
            shape = str(image.size).encode()
            # Aynı pikselleri ve tam yatay aynalarını tek grupta tutar.
            pixel_hash = min(
                hashlib.sha256(shape + image.tobytes()).hexdigest(),
                hashlib.sha256(shape + ImageOps.mirror(image).tobytes()).hexdigest(),
            )
            stat = path.stat()
            keys = [
                ("file", file_hash),
                ("pixel", pixel_hash),
                ("inode", stat.st_dev, stat.st_ino),
                ("path", str(resolved).casefold()),
            ]
            match = re.fullmatch(r"(real|fake)_(\d+)_aug_\d+", path.stem, re.IGNORECASE)
            if match:
                augmentation_count += 1
                keys.append(("augmentation", label, match[1].lower(), match[2]))
            for key in keys:
                if key in seen:
                    union(relative, seen[key])
                    if key[0] in {"file", "pixel", "inode", "path"}:
                        duplicates.append(
                            {"kind": key[0], "first": seen[key], "second": relative}
                        )
                else:
                    seen[key] = relative
            records.append(
                {
                    "path": relative,
                    "label": label,
                    "sha256": file_hash,
                    "pixel_sha256": pixel_hash,
                    "width": image.width,
                    "height": image.height,
                }
            )
        except (OSError, ValueError, Image.DecompressionBombError) as error:
            errors.append({"path": relative, "error": str(error)})
        if progress and index % 1000 == 0:
            print(f"Denetlendi: {index}/{len(samples)}", flush=True)
    groups = {path: find(path) for path, _ in samples}
    counts = {
        name: sum(label == index for _, label in samples)
        for index, name in enumerate(CLASS_NAMES)
    }
    report = {
        "seed": seed,
        "counts": counts,
        "total": len(samples),
        "records": records,
        "errors": errors,
        "path_errors": path_errors,
        "duplicates": duplicates,
        "augmentation_images": augmentation_count,
        "groups": groups,
        "source_identity_verified": False,
        "source_limitation": "Augmentation aileleri biliniyor; orijinal kaynak/kişi eşlemesi sağlanmadı.",
    }
    csv_path = root / "dataset.csv"
    if csv_path.exists():
        import pandas as pd

        frame = pd.read_csv(csv_path)
        if {"path", "label"}.issubset(frame.columns):
            csv_samples = [
                (
                    f"{row.label}/{str(row.path).replace(chr(92), '/').split('/')[-1]}",
                    CLASS_TO_IDX.get(row.label, -1),
                )
                for row in frame.itertuples()
            ]
            report["csv"] = {
                "rows": len(frame),
                "duplicate_paths": int(frame.path.duplicated().sum()),
                "legacy_absolute_paths": int(
                    frame.path.astype(str).str.startswith("/kaggle/").sum()
                ),
                "local_mapping_matches": sorted(csv_samples) == sorted(samples),
            }
        else:
            report["csv"] = {"error": "CSV path ve label sütunlarını içermiyor."}
    if not errors and not path_errors:
        try:
            splits = split_samples(samples, seed, groups)
            report["splits"] = splits
            memberships = {}
            overlaps = []
            for name, subset in splits.items():
                for path, _ in subset:
                    group = groups[path]
                    if group in memberships and memberships[group] != name:
                        overlaps.append(group)
                    memberships[group] = name
            report["cross_split_groups"] = sorted(set(overlaps))
            report["split_counts"] = {
                name: {
                    "total": len(subset),
                    **{
                        class_name: sum(label == index for _, label in subset)
                        for index, class_name in enumerate(CLASS_NAMES)
                    },
                }
                for name, subset in splits.items()
            }
        except ValueError as error:
            report["split_error"] = str(error)
    signature = [
        (record["path"], record["label"], record["sha256"]) for record in records
    ]
    report["fingerprint"] = hashlib.sha256(json.dumps(signature).encode()).hexdigest()
    return report


def image_transform(training: bool = False) -> transforms.Compose:
    steps = [transforms.Resize((IMAGE_SIZE, IMAGE_SIZE), antialias=True)]
    if training:
        steps.append(transforms.RandomHorizontalFlip())
    steps.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )
    return transforms.Compose(steps)


def read_image(path: str | Path) -> Image.Image:
    try:
        with Image.open(path) as image:
            return ImageOps.exif_transpose(image).convert("RGB")
    except (OSError, ValueError) as error:
        raise ValueError(f"Görüntü okunamadı: {path}") from error


class FaceDataset(Dataset):
    def __init__(self, root: str | Path, samples: list[Sample], training: bool = False):
        self.root = Path(root)
        self.samples = samples
        self.transform = image_transform(training)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        relative_path, label = self.samples[index]
        return self.transform(read_image(self.root / relative_path)), label


def make_loader(
    dataset: FaceDataset,
    batch_size: int = 8,
    workers: int = 0,
    shuffle: bool = False,
    pin_memory: bool = False,
    seed: int = SEED,
) -> DataLoader:
    if batch_size < 1 or workers < 0:
        raise ValueError("Batch size pozitif, workers sıfır veya pozitif olmalı.")
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=pin_memory,
        worker_init_fn=seed_worker,
        generator=torch.Generator().manual_seed(seed),
        persistent_workers=workers > 0,
    )
