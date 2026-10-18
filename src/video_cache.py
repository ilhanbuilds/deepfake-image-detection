"""Read-only complete train/val cache audit; never decodes or reads official test videos."""

import argparse
import json
from collections import Counter
from pathlib import Path
from time import perf_counter
from uuid import uuid4

from src.video_manifest import DEFAULT_MANIFEST, digest_json, load_manifest
from src.video_model import VideoClassifier
from src.video_pipeline import FacePipeline, FeatureCache, VideoSettings, file_sha256


def audit_feature_cache(manifest: dict, cache: FeatureCache) -> dict:
    started = perf_counter()
    root = Path(manifest["dataset_root"])
    splits = {name: manifest["splits"][name] for name in ("train", "val")}
    seen_paths, seen_families = {}, {}
    expected = set()
    for name, records in splits.items():
        for record in records:
            path, family = record["path"], record["family"]
            if path in seen_paths or (
                family in seen_families and seen_families[family] != name
            ):
                raise ValueError("Train/validation video veya kaynak-aile çakışması.")
            seen_paths[path], seen_families[family] = name, name
            expected.add(cache.path(record))
    actual = set(cache.directory.glob("*.pt"))
    missing, unexpected = sorted(expected - actual), sorted(actual - expected)
    if missing or unexpected:
        raise ValueError(
            f"Cache eksik/beklenmeyen kayıt: missing={len(missing)}, unexpected={len(unexpected)}; ilk={str((missing or unexpected)[0])}"
        )
    report, signature = {}, []
    strict_codes = manifest.get("schema") == "celeb_df_identity_components_v1"
    source_memberships = {}
    for name, records in splits.items():
        accepted, rejected, histogram = [], [], Counter()
        for index, record in enumerate(records, 1):
            path = cache.path(record)
            if path.is_symlink():
                raise ValueError(f"Cache symlink kabul edilmez: {path}")
            item = cache.read(record, root)
            if strict_codes:
                source = root / record["path"]
                stat = source.stat()
                source_hash = cache.verified_sources[
                    (str(source.resolve()), stat.st_size, stat.st_mtime_ns)
                ]
                source_memberships.setdefault(source_hash, {}).setdefault(
                    name, []
                ).append(record["path"])
            signature.append((name, record, file_sha256(path)))
            if item["values"] is None:
                rejected.append(item["report"])
            else:
                accepted.append(record)
                histogram[len(item["values"])] += 1
            if index % 500 == 0:
                print(f"Cache audit {name}: {index}/{len(records)}", flush=True)
        report[name] = dict(
            total=len(records),
            accepted=len(accepted),
            rejected=len(rejected),
            accepted_counts={
                class_name: sum(r["label"] == label for r in accepted)
                for label, class_name in enumerate(("REAL", "FAKE"))
            },
            valid_frame_histogram=dict(sorted(histogram.items())),
            accepted_records=accepted,
            rejected_reports=rejected,
        )
        if not all(report[name]["accepted_counts"].values()):
            raise ValueError(f"Kabul edilmiş {name} cache iki sınıfı içermeli.")
    result = dict(
        schema="video_train_val_cache_audit_v1",
        manifest_fingerprint=manifest["fingerprint"],
        cache_fingerprint=digest_json(signature),
        cache_signature=cache.signature,
        cache_directory=str(cache.directory.resolve()),
        splits=report,
        cross_split_paths=[],
        cross_split_families=[],
        official_test_accessed=False,
        elapsed_seconds=perf_counter() - started,
    )
    if strict_codes:
        code_sets = {
            name: {i for r in records for i in r["identities"]}
            for name, records in splits.items()
        }
        duplicate_content = [
            dict(sha256=sha, splits=members)
            for sha, members in source_memberships.items()
            if len(members) > 1
        ]
        if code_sets["train"] & code_sets["val"] or duplicate_content:
            raise ValueError(
                "Identity split train/validation kod veya aynı MP4 içerik çakışması."
            )
        result.update(
            cross_split_codes=[],
            cross_split_identical_mp4=[],
            content_check="SHA-256 of complete train/val MP4 bytes; re-encoded/partial/shared-source duplicates unverified",
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Yalnız mevcut train/validation cache bütünlüğü; eğitim veya decode yok."
    )
    parser.add_argument("--check-cache", action="store_true", required=True)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--cache-dir", type=Path, default=Path("outputs/video_v1/cache")
    )
    parser.add_argument("--image-size", type=int, choices=(160, 192), default=160)
    args = parser.parse_args()
    manifest = load_manifest(args.manifest)
    pipeline = FacePipeline(VideoSettings(image_size=args.image_size))
    model = VideoClassifier(pretrained=True).eval()
    cache = FeatureCache(
        args.cache_dir, pipeline.signature, model.backbone_fingerprint()
    )
    report = audit_feature_cache(manifest, cache)
    directory = Path("outputs/video_v1/cache_audits") / uuid4().hex
    directory.mkdir(parents=True, exist_ok=False)
    (directory / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                name: {
                    k: v
                    for k, v in subset.items()
                    if k not in {"accepted_records", "rejected_reports"}
                }
                for name, subset in report["splits"].items()
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    print(f"Denetim: {directory / 'report.json'}; eğitim başlatılmadı.")


if __name__ == "__main__":
    main()
