"""Accepted-video balancing, replacement coverage and source diagnostics."""

from collections import Counter
from statistics import mean, median

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from src.video_pipeline import video_collate


def make_balanced_video_loader(dataset, records, seed=42):
    if len(dataset) != len(records) or not records:
        raise ValueError("Sampler yalnız kabul edilmiş train roster kullanmalı.")
    if getattr(dataset, "records", records) != records:
        raise ValueError("Dataset sırası sampler train roster ile eşleşmiyor.")
    counts = Counter(record["label"] for record in records)
    if set(counts) != {0, 1} or len({r["path"] for r in records}) != len(records):
        raise ValueError("Train roster benzersiz videolar ve iki sınıf içermeli.")
    generator = torch.Generator().manual_seed(seed)
    sampler = WeightedRandomSampler(
        [1 / counts[record["label"]] for record in records],
        num_samples=len(records),
        replacement=True,
        generator=generator,
    )
    # One shared generator: existing capture_rng/restore_rng also restores sampling.
    return DataLoader(
        dataset,
        batch_size=1,
        sampler=sampler,
        num_workers=0,
        collate_fn=video_collate,
        generator=generator,
    )


def sampling_summary(records, draws):
    roster = {r["path"]: r["label"] for r in records}
    if len(roster) != len(records) or any(
        r["path"] not in roster or roster[r["path"]] != r["label"] for r in draws
    ):
        raise ValueError("Örneklenen video kabul edilmiş train roster dışında.")
    by_class = {}
    for label, name in enumerate(("REAL", "FAKE")):
        counts = Counter(r["path"] for r in draws if r["label"] == label)
        available = sum(value == label for value in roster.values())
        sampled = sum(counts.values())
        by_class[name] = dict(
            available_videos=available,
            sampled_draws=sampled,
            sampling_fraction=sampled / len(draws) if draws else 0.0,
            unique_videos=len(counts),
            unique_coverage=len(counts) / available if available else None,
            repeated_videos=sum(count > 1 for count in counts.values()),
            repeat_draws=sampled - len(counts),
            maximum_multiplicity=max(counts.values(), default=0),
            draws_per_video=dict(sorted(counts.items())),
        )
    return dict(
        scope="actual_successful_optimizer_draws",
        total_draws=len(draws),
        unique_videos=sum(c["unique_videos"] for c in by_class.values()),
        by_class=by_class,
    )


def gradient_summary(rows):
    result = {}
    for label, name in enumerate(("REAL", "FAKE")):
        group = [r for r in rows if r["label"] == label]
        before = [r["pre_clip"] for r in group]
        after = [r["post_clip"] for r in group]
        result[name] = dict(
            optimizer_steps=len(group),
            clipped_steps=sum(norm > 1.0 for norm in before),
            clipping_rate=sum(norm > 1.0 for norm in before) / len(group)
            if group
            else None,
            norm_pre_clip=dict(
                mean=mean(before), median=median(before), max=max(before)
            )
            if group
            else None,
            norm_post_clip=dict(mean=mean(after), median=median(after), max=max(after))
            if group
            else None,
        )
    return result


def source_metrics(predictions):
    """Single-class folders have no standalone AUC; recall and scores are defined."""
    result = {}
    for source in ("Celeb-real", "YouTube-real", "Celeb-synthesis"):
        rows = [r for r in predictions if r["path"].split("/")[0] == source]
        scores = [r["fake_probability"] for r in rows]
        correct = sum(r["target"] == r["prediction"] for r in rows)
        result[source] = dict(
            support=len(rows),
            errors=len(rows) - correct,
            recall=correct / len(rows) if rows else None,
            predicted_fake=sum(r["prediction"] for r in rows),
            fake_score_mean=mean(scores) if scores else None,
            fake_score_median=median(scores) if scores else None,
        )
    return result


def comparison_metrics(evaluated):
    metrics = evaluated["metrics"]
    sources = source_metrics(evaluated["predictions"])
    return dict(
        **{
            k: metrics[k]
            for k in ("macro_f1", "roc_auc", "balanced_accuracy", "accuracy")
        },
        real_recall=metrics["per_class"]["REAL"]["recall"],
        fake_recall=metrics["per_class"]["FAKE"]["recall"],
        celeb_real_recall=sources["Celeb-real"]["recall"],
        youtube_real_recall=sources["YouTube-real"]["recall"],
        sources=sources,
        threshold=evaluated["threshold"],
    )
