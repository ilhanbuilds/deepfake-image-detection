"""Filename-code connected components; immutable official holdout, no decoding."""

import argparse
import copy
import csv
import itertools
import json
import math
import random
from collections import Counter
from pathlib import Path

from src.video_manifest import DEFAULT_MANIFEST, digest_json, family_info

SCHEMA = "celeb_df_identity_components_v1"
LIMITATION = (
    "Yalnız filename kodları/ilişkileri ayrı. Gerçek kişi, donor kaynak klipleri ve "
    "YouTube kişi/kanal/session metadata'sı doğrulanmadı; gerçek identity-disjoint garantisi yok. "
    "Resmi test/karantina eski protokolle sabit; test kimliklerinin train/val ile ayrı olduğu iddia edilmez."
)


def identity_components(records):
    adjacency = {}
    for record in records:
        family, identities = family_info(record["path"])
        if family != record["family"] or identities != record["identities"]:
            raise ValueError("Filename identity/family metadata uyumsuz.")
        for identity in identities:
            adjacency.setdefault(identity, set()).update(identities)
    remaining, components = set(adjacency), []
    while remaining:
        stack, seen = [min(remaining)], set()
        while stack:
            identity = stack.pop()
            if identity in seen:
                continue
            seen.add(identity)
            stack.extend(sorted(adjacency[identity] - seen))
        remaining -= seen
        components.append(sorted(seen))
    return components


def partition_components(parent, fraction=0.2, seed=42):
    if not 0 < fraction < 1:
        raise ValueError("Validation oranı 0–1 arasında olmalı.")
    # Protected records contribute filename relationships ONLY, never videos or metrics.
    all_records = [
        r
        for name in ("train", "val", "test", "quarantine")
        for r in parent["splits"][name]
    ]
    components = identity_components(all_records)
    if not 2 <= len(components) <= 16:
        raise ValueError(
            "Bileşenleri bölmeden güvenli iki sınıflı ayrım için 2–16 bileşen gerekli; yeni split oluşturulmadı."
        )
    component_for = {
        identity: index for index, ids in enumerate(components) for identity in ids
    }
    eligible = sorted(
        parent["splits"]["train"] + parent["splits"]["val"], key=lambda r: r["path"]
    )
    celeb = [r for r in eligible if r["identities"]]
    youtube = [r for r in eligible if not r["identities"]]
    for record in celeb:
        if len({component_for[i] for i in record["identities"]}) != 1:
            raise ValueError("Manipülasyon iki graph bileşenine yayılıyor.")
    youtube = list(youtube)
    random.Random(seed).shuffle(youtube)
    youtube_val = {r["path"] for r in youtube[: round(len(youtube) * fraction)]}
    totals = Counter(r["label"] for r in eligible)
    best = None
    # Choose using inventory class proportions ONLY; no scores, cache quality or labels at test.
    for size in range(1, len(components)):
        for chosen in itertools.combinations(range(len(components)), size):
            selected = set(chosen)
            val = [
                r
                for r in eligible
                if (r["identities"] and component_for[r["identities"][0]] in selected)
                or r["path"] in youtube_val
            ]
            val_paths = {r["path"] for r in val}
            train = [r for r in eligible if r["path"] not in val_paths]
            if any({r["label"] for r in subset} != {0, 1} for subset in (train, val)):
                continue
            counts = Counter(r["label"] for r in val)
            error = sum(
                abs(counts[label] / totals[label] - fraction) for label in (0, 1)
            )
            rank = (error, abs(len(val) / len(eligible) - fraction), chosen)
            if best is None or rank < best[0]:
                best = rank, train, val, chosen
    if best is None:
        raise ValueError(
            "Kod bileşenleri bölünmeden iki sınıflı train/validation oluşturulamıyor."
        )
    _, train, val, chosen = best
    return (
        train,
        val,
        dict(
            algorithm="all_filename_code_connected_components_v1",
            graph_scope="all_manifest_filename_metadata_only",
            component_identities=components,
            validation_components=list(chosen),
            selection="inventory_class_fraction_error_only_no_model_metrics",
            youtube_policy="seed42_video_disjoint_identity_unknown",
            dropped_videos=0,
        ),
    )


def build_identity_manifest(parent, parent_path, fraction=0.2, seed=42):
    if parent.get("schema") != "celeb_df_video_v1":
        raise ValueError(
            "Identity manifest yalnız korunmuş orijinal V1 manifestten türetilir."
        )
    train, val, policy = partition_components(parent, fraction, seed)
    report = copy.deepcopy(parent)
    report.pop("fingerprint")
    report.update(
        schema=SCHEMA,
        parent_manifest=str(Path(parent_path).resolve()),
        parent_fingerprint=parent["fingerprint"],
        seed=seed,
        validation_fraction=fraction,
        split_policy=policy,
        source_identity_verified=False,
        filename_identity_disjoint=True,
        limitation=LIMITATION,
    )
    report["splits"]["train"], report["splits"]["val"] = train, val
    report["split_counts"] = {
        name: dict(
            total=len(rows),
            REAL=sum(r["label"] == 0 for r in rows),
            FAKE=sum(r["label"] == 1 for r in rows),
        )
        for name, rows in report["splits"].items()
    }
    identity_sets = {
        name: {i for r in report["splits"][name] for i in r["identities"]}
        for name in ("train", "val", "test")
    }
    report["identity_overlap_counts"] = {
        f"{a}_{b}": len(identity_sets[a] & identity_sets[b])
        for a, b in (("train", "val"), ("train", "test"), ("val", "test"))
    }
    if report["identity_overlap_counts"]["train_val"]:
        raise ValueError("Yeni train/val filename identity çakışması.")
    report["fingerprint"] = digest_json(report)
    return report


def load_identity_manifest(path, report, verify_inventory=True):
    from src.video_manifest import load_manifest

    unsigned = {k: v for k, v in report.items() if k != "fingerprint"}
    if digest_json(unsigned) != report.get("fingerprint"):
        raise ValueError("Identity manifest bütünlüğü geçersiz.")
    parent_path = Path(report["parent_manifest"])
    if parent_path.resolve() == Path(path).resolve():
        raise ValueError("Identity manifest kendisine referans veremez.")
    parent_header = json.loads(parent_path.read_text(encoding="utf-8"))
    if parent_header.get("schema") != "celeb_df_video_v1":
        raise ValueError("Identity parent yalnız orijinal V1 manifest olabilir.")
    parent = load_manifest(parent_path, verify_inventory=verify_inventory)
    expected = build_identity_manifest(
        parent, parent_path, report["validation_fraction"], report["seed"]
    )
    if expected != report:
        raise ValueError("Identity split/parent/official holdout politikası değişmiş.")
    return report


def role_codes(record):
    folder = record["path"].split("/")[0]
    parts = Path(record["path"]).stem.split("_")
    if folder == "Celeb-real":
        return {"real_code": parts[0]}
    if folder == "Celeb-synthesis":
        return {"fake_first_code": parts[0], "fake_second_code": parts[1]}
    return {}


def overlap_details(parent):
    role_counts, examples = {}, {}
    for name in ("train", "val"):
        counts, samples = {}, {}
        for record in parent["splits"][name]:
            for role, code in role_codes(record).items():
                counts.setdefault(code, Counter())[role] += 1
                samples.setdefault(code, {}).setdefault(role, [])
                if len(samples[code][role]) < 3:
                    samples[code][role].append(record["path"])
        role_counts[name], examples[name] = counts, samples
    shared = sorted(
        role_counts["train"].keys() & role_counts["val"].keys(),
        key=lambda x: int(x[2:]),
    )
    return [
        dict(
            code=code,
            train=dict(role_counts["train"][code]),
            validation=dict(role_counts["val"][code]),
            examples={name: examples[name][code] for name in ("train", "val")},
        )
        for code in shared
    ]


def split_distribution(manifest):
    report = {}
    for name in ("train", "val"):
        rows = manifest["splits"][name]
        report[name] = dict(
            **manifest["split_counts"][name],
            folders=dict(Counter(r["path"].split("/")[0] for r in rows)),
            families=len({r["family"] for r in rows}),
            codes=sorted(
                {i for r in rows for i in r["identities"]}, key=lambda x: int(x[2:])
            ),
        )
    return report


def main():
    from src.video_cache import audit_feature_cache
    from src.video_manifest import load_manifest
    from src.video_model import VideoClassifier
    from src.video_pipeline import FacePipeline, FeatureCache, VideoSettings
    from src.video_train import training_class_balance

    parser = argparse.ArgumentParser(
        description="Metadata-only identity-code split; no training, decoding or preprocessing."
    )
    parser.add_argument("--prepare-split", action="store_true", required=True)
    parser.add_argument("--parent-manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/video_identity_v1")
    )
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument(
        "--cache-dir", type=Path, default=Path("outputs/video_v1/cache")
    )
    args = parser.parse_args()
    if not math.isfinite(args.validation_fraction):
        raise ValueError("Validation oranı sonlu olmalı.")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    for filename in (
        "manifest.json",
        "split_report.json",
        "cache_audit.json",
        "shared_codes.csv",
    ):
        if (output / filename).exists():
            raise FileExistsError(
                "Mevcut identity raporu korunuyor; yeni output-dir kullanın."
            )
    parent = load_manifest(args.parent_manifest)
    manifest = build_identity_manifest(
        parent, args.parent_manifest, args.validation_fraction
    )
    pipeline = FacePipeline(VideoSettings())
    model = VideoClassifier(pretrained=True)
    cache = FeatureCache(
        args.cache_dir, pipeline.signature, model.backbone_fingerprint()
    )
    cache_audit = audit_feature_cache(manifest, cache)
    balance = training_class_balance(cache_audit["splits"]["train"]["accepted_records"])
    shared = overlap_details(parent)
    report = dict(
        parent_manifest_fingerprint=parent["fingerprint"],
        new_manifest_fingerprint=manifest["fingerprint"],
        old_shared_codes=shared,
        old_shared_code_count=len(shared),
        graph=manifest["split_policy"],
        distribution=split_distribution(manifest),
        accepted_cache_distribution={
            name: {
                k: v
                for k, v in rows.items()
                if k not in {"accepted_records", "rejected_reports"}
            }
            for name, rows in cache_audit["splits"].items()
        },
        training_class_balance=balance,
        new_identity_overlap_counts=manifest["identity_overlap_counts"],
        cross_split_families=cache_audit["cross_split_families"],
        official_test_unchanged=manifest["splits"]["test"] == parent["splits"]["test"],
        quarantine_unchanged=manifest["splits"]["quarantine"]
        == parent["splits"]["quarantine"],
        record_cache_keys_unchanged=True,
        feature_cache_recomputed=False,
        old_checkpoints_valid_for_resume_on_new_split=False,
        clean_initialization_required="ImageNet backbone plus random GRU/head; old V1/V2/benchmark learned weights exposed new validation videos",
        source_identity_verified=False,
        limitation=LIMITATION,
        decoded_videos=0,
        training_started=False,
        sources=[
            "https://github.com/yuezunli/celeb-deepfakeforensics",
            "https://arxiv.org/html/1909.12962",
        ],
    )
    for filename, value in (
        ("manifest.json", manifest),
        ("cache_audit.json", cache_audit),
        ("split_report.json", report),
    ):
        with (output / filename).open("x", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)
    with (output / "shared_codes.csv").open(
        "x", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "code",
                *[
                    f"{split}_{role}"
                    for split in ("train", "validation")
                    for role in ("real_code", "fake_first_code", "fake_second_code")
                ],
            ]
        )
        for row in shared:
            writer.writerow(
                [
                    row["code"],
                    *[
                        row[split].get(role, 0)
                        for split in ("train", "validation")
                        for role in ("real_code", "fake_first_code", "fake_second_code")
                    ],
                ]
            )
    print(
        json.dumps(
            {
                k: report[k]
                for k in (
                    "distribution",
                    "accepted_cache_distribution",
                    "training_class_balance",
                    "new_identity_overlap_counts",
                    "official_test_unchanged",
                    "quarantine_unchanged",
                )
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
