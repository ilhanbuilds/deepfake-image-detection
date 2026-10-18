"""Celeb-DF-v2 video inventory, official holdout and filename-family splits."""

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath

from sklearn.model_selection import GroupShuffleSplit

FOLDERS = {"Celeb-real": 0, "YouTube-real": 0, "Celeb-synthesis": 1}
DEFAULT_ROOT = Path("data/video/celeb_df_v2")
DEFAULT_MANIFEST = Path("outputs/video_v1/manifest.json")


def digest_json(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def family_info(relative: str) -> tuple[str, list[str]]:
    folder, filename = relative.split("/")
    stem = Path(filename).stem
    if folder == "Celeb-real":
        match = re.fullmatch(r"(id\d+)_(\d{4})", stem)
        if match:
            return f"celeb:{match[1]}:{match[2]}", [match[1]]
    elif folder == "Celeb-synthesis":
        match = re.fullmatch(r"(id\d+)_(id\d+)_(\d{4})", stem)
        if match:
            # Filename convention only: first identity + clip index = carrier video.
            # Donor identity is recorded, not asserted to be identity-disjoint.
            return f"celeb:{match[1]}:{match[3]}", sorted({match[1], match[2]})
    elif folder == "YouTube-real" and re.fullmatch(r"\d{5}", stem):
        return f"youtube:{stem}", []
    raise ValueError(f"Beklenmeyen video adı; metadata gerekli: {relative}")


def parse_official(root: Path, records: dict[str, dict]) -> dict[str, int]:
    """Observed local format: integer (1 REAL / 0 FAKE), space, POSIX mp4 path."""
    result = {}
    for number, line in enumerate(
        (root / "List_of_testing_videos.txt").read_text(encoding="utf-8").splitlines(),
        1,
    ):
        match = re.fullmatch(r"([01]) ([^\s]+\.mp4)", line)
        if not match:
            raise ValueError(f"Resmi test satırı {number} formatı geçersiz: {line!r}")
        relative = match[2]
        path = PurePosixPath(relative)
        if path.is_absolute() or len(path.parts) != 2 or ".." in path.parts:
            raise ValueError(f"Güvensiz test yolu: {relative}")
        if relative not in records:
            raise ValueError(f"Resmi test videosu bulunamadı: {relative}")
        label = 1 - int(match[1])
        if records[relative]["label"] != label or relative in result:
            raise ValueError(f"Test etiketi/tekrar hatası: {relative}")
        result[relative] = label
    if not result or set(result.values()) != {0, 1}:
        raise ValueError("Resmi test listesi iki sınıfı içermeli.")
    return result


def grouped_split(records: list[dict], fraction: float, seed: int) -> tuple[list, list]:
    if not 0 < fraction < 1 or len({r["family"] for r in records}) < 2:
        raise ValueError("Validation oranı/grup sayısı geçersiz.")
    splitter = GroupShuffleSplit(n_splits=128, test_size=fraction, random_state=seed)
    best = None
    best_error = float("inf")
    totals = Counter(r["label"] for r in records)
    for train_idx, val_idx in splitter.split(
        records, groups=[r["family"] for r in records]
    ):
        train = [records[i] for i in train_idx]
        val = [records[i] for i in val_idx]
        if any({r["label"] for r in subset} != {0, 1} for subset in (train, val)):
            continue
        counts = Counter(r["label"] for r in val)
        error = sum(abs(counts[label] / totals[label] - fraction) for label in (0, 1))
        if error < best_error:
            best, best_error = (train, val), error
    if best is None:
        raise ValueError("Karantina sonrası iki sınıflı train/validation grupları yok.")
    return best


def audit(root: Path, fraction: float = 0.2, seed: int = 42) -> dict:
    root = root.resolve()
    records = {}
    for folder, label in FOLDERS.items():
        directory = root / folder
        if not directory.is_dir():
            raise ValueError(f"Dataset klasörü eksik: {directory}")
        paths = sorted(directory.iterdir())
        if not paths:
            raise ValueError(f"Boş video klasörü: {directory}")
        for path in paths:
            if path.is_symlink() or not path.resolve().is_relative_to(root):
                raise ValueError(f"Güvensiz dataset yolu: {path}")
            if not path.is_file() or path.suffix != ".mp4":
                raise ValueError(f"Beklenmeyen dataset girdisi: {path}")
            relative = path.relative_to(root).as_posix()
            family, identities = family_info(relative)
            stat = path.stat()
            if not stat.st_size:
                raise ValueError(f"Boş video: {relative}")
            records[relative] = dict(
                path=relative,
                label=label,
                family=family,
                identities=identities,
                size=stat.st_size,
                mtime_ns=stat.st_mtime_ns,
            )
    for record in records.values():
        if record["label"] == 1:
            _, identity, index = record["family"].split(":")
            if f"Celeb-real/{identity}_{index}.mp4" not in records:
                raise ValueError(f"Dosya adı kaynak adayı bulunamadı: {record['path']}")
    official = parse_official(root, records)
    test_families = {records[path]["family"] for path in official}
    test, quarantine, remaining = [], [], []
    for relative, record in sorted(records.items()):
        if relative in official:
            test.append(record)
        elif record["family"] in test_families:
            quarantine.append(record)
        else:
            remaining.append(record)
    train, val = grouped_split(remaining, fraction, seed)
    splits = dict(train=train, val=val, test=test, quarantine=quarantine)
    memberships = defaultdict(set)
    identity_sets = {}
    for name in ("train", "val", "test"):
        identity_sets[name] = {
            identity for r in splits[name] for identity in r["identities"]
        }
        for record in splits[name]:
            memberships[record["family"]].add(name)
    overlaps = sorted(f for f, names in memberships.items() if len(names) > 1)
    if overlaps:
        raise ValueError(f"Kaynak aile çakışması: {overlaps}")
    report = dict(
        schema="celeb_df_video_v1",
        dataset_root=str(root),
        seed=seed,
        validation_fraction=fraction,
        official_label_mapping={"1": "REAL", "0": "FAKE"},
        class_names=["REAL", "FAKE"],
        official_sha256=hashlib.sha256(
            (root / "List_of_testing_videos.txt").read_bytes()
        ).hexdigest(),
        folder_counts={
            folder: sum(r["path"].startswith(folder + "/") for r in records.values())
            for folder in FOLDERS
        },
        splits=splits,
        split_counts={
            name: dict(
                total=len(subset),
                REAL=sum(r["label"] == 0 for r in subset),
                FAKE=sum(r["label"] == 1 for r in subset),
            )
            for name, subset in splits.items()
        },
        cross_split_families=overlaps,
        test_related_families_quarantined=len({r["family"] for r in quarantine}),
        identity_overlap_counts={
            f"{a}_{b}": len(identity_sets[a] & identity_sets[b])
            for a, b in (("train", "val"), ("train", "test"), ("val", "test"))
        },
        source_identity_verified=False,
        limitation="Aileler dosya adı first-id + clip-index çıkarımıdır; yerel kaynak/identity metadata yok. Donor kimliği ve YouTube kaynak ilişkileri doğrulanmadı. Identity-disjoint değildir. Video içerik hash/decode denetimi henüz yapılmadı.",
    )
    report["fingerprint"] = digest_json(report)
    return report


def load_manifest(path: Path, verify_inventory: bool = True) -> dict:
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("schema") == "celeb_df_identity_components_v1":
        from src.video_identity_split import load_identity_manifest

        return load_identity_manifest(path, report, verify_inventory)
    fingerprint = report.pop("fingerprint")
    if (
        report.get("schema") != "celeb_df_video_v1"
        or digest_json(report) != fingerprint
    ):
        raise ValueError("Video manifest bütünlüğü/şeması geçersiz.")
    report["fingerprint"] = fingerprint
    if verify_inventory:
        current = audit(
            Path(report["dataset_root"]), report["validation_fraction"], report["seed"]
        )
        if current["fingerprint"] != fingerprint:
            raise ValueError("Dataset/resmi test/split değişti; yeni manifest gerekli.")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Yalnız video inventory/split; decode/eğitim yapmaz."
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    args = parser.parse_args()
    report = audit(args.data_dir, args.validation_fraction)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "folder_counts",
                    "split_counts",
                    "cross_split_families",
                    "identity_overlap_counts",
                    "limitation",
                )
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
