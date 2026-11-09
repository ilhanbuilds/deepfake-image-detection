"""Resolve historical Windows provenance paths inside the current project root."""

from pathlib import Path, PureWindowsPath
import hashlib

IMPLEMENTATIONS = frozenset(
    {
        "video_forensic.py",
        "video_forensic_train.py",
        "video_cache.py",
        "train.py",
        "video_evaluate.py",
        "video_sbi12.py",
        "video_frame_study.py",
        "video_spatial_evaluate.py",
        "video_train.py",
    }
)


def verify_implementations(recorded, root):
    """Rebase only known source identities; retain every original byte hash."""
    root = Path(root).resolve()
    seen = set()
    for historical, expected in recorded.items():
        path = PureWindowsPath(historical)
        if (
            path.name not in IMPLEMENTATIONS
            or path.parent.name != "src"
            or ".." in path.parts
            or (not path.is_absolute() and len(path.parts) != 2)
            or path.name in seen
        ):
            raise ValueError("Unknown or ambiguous implementation provenance path.")
        current = root / "src" / path.name
        if current.is_symlink() or current.resolve().parent != root / "src":
            raise ValueError("Implementation source escaped the project root.")
        with current.open("rb") as source:
            actual = hashlib.file_digest(source, "sha256").hexdigest()
        if actual != expected:
            raise ValueError(f"Implementation checksum mismatch: {path.name}")
        seen.add(path.name)
    if seen != IMPLEMENTATIONS:
        raise ValueError("Incomplete implementation provenance.")
