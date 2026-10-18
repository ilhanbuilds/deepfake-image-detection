"""Hazır train/valid ayrımlarını korur; test klasörüne erişmez."""

import hashlib
from pathlib import Path

from torchvision.datasets import ImageFolder

from src.dataset import image_transform, read_image


def resolve_data_root(root):
    root = Path(root).resolve()
    if not root.is_dir() and root.name == "real_vs_fake":
        alternative = root.with_name("real-vs-fake")
        if alternative.is_dir():
            print(
                f"Dataset yolu: {alternative} (mevcut klasör adı kullanılıyor)",
                flush=True,
            )
            return alternative
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset bulunamadı: {root}")
    return root


class PresplitFaceDataset(ImageFolder):
    """ImageFolder örnekleri ve targets dahil REAL=0 / FAKE=1 eşlemesi."""

    def __init__(self, root, split):
        if split not in ("train", "valid"):
            raise ValueError("Training loader yalnızca train veya valid okuyabilir.")
        self.split = split
        super().__init__(
            str(resolve_data_root(root) / split),
            transform=image_transform(training=split == "train"),
            loader=read_image,
        )

    def find_classes(self, directory):
        actual = {p.name for p in Path(directory).iterdir() if p.is_dir()}
        if actual != {"real", "fake"}:
            raise ValueError(f"Beklenen sınıf klasörleri real/fake; bulunan: {actual}")
        return ["real", "fake"], {"real": 0, "fake": 1}


def split_counts(datasets):
    return {
        name: {
            "REAL": dataset.targets.count(0),
            "FAKE": dataset.targets.count(1),
            "total": len(dataset),
        }
        for name, dataset in datasets.items()
    }


def dataset_fingerprint(datasets):
    """Yalnızca training/validation içerikleri; resume değişiklikleri reddeder."""
    fingerprint = hashlib.sha256()
    for name, dataset in sorted(datasets.items()):
        for path, label in dataset.samples:
            path = Path(path)
            digest = hashlib.sha256()
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            fingerprint.update(
                f"{name}:{path.relative_to(dataset.root).as_posix()}:{label}:".encode()
            )
            fingerprint.update(digest.digest())
    return fingerprint.hexdigest()
