"""Train-only class weighting and bounded, reproducible preprocessing benchmarks."""

import math

import pytest
import torch

from src.video_preprocess import (
    estimate_preprocessing,
    make_parser,
    select_balanced_sample,
)
from src.video_train import training_class_balance, weighted_video_loss


def train_records():
    return [dict(path=f"real/{i:04d}.mp4", label=0) for i in range(438)] + [
        dict(path=f"fake/{i:04d}.mp4", label=1) for i in range(1892)
    ]


def test_balanced_weights_use_train_counts_and_equal_class_contributions():
    report = training_class_balance(train_records())
    assert report["counts"] == {"REAL": 438, "FAKE": 1892}
    assert report["class_weights"]["REAL"] == pytest.approx(2330 / (2 * 438))
    assert report["class_weights"]["FAKE"] == pytest.approx(2330 / (2 * 1892))
    assert report["source"] == "accepted_train_cache_only"
    assert report["class_weights"]["REAL"] * 438 == pytest.approx(
        report["class_weights"]["FAKE"] * 1892
    )
    labels = torch.tensor([r["label"] for r in train_records()], dtype=torch.float32)
    weights = torch.tensor(
        [report["class_weights"]["REAL"], report["class_weights"]["FAKE"]]
    )
    assert weighted_video_loss(
        torch.zeros(2330), labels, weights
    ).item() == pytest.approx(math.log(2))


def test_weighted_loss_emphasizes_minority_real_per_video():
    report = training_class_balance(train_records())
    weights = torch.tensor(
        [report["class_weights"]["REAL"], report["class_weights"]["FAKE"]]
    )
    logits = torch.zeros(2, requires_grad=True)
    loss = weighted_video_loss(logits, torch.tensor([0.0, 1.0]), weights)
    loss.backward()
    assert logits.grad[0].abs() / logits.grad[1].abs() == pytest.approx(1892 / 438)
    with pytest.raises(ValueError, match="Her video"):
        weighted_video_loss(torch.zeros(2, 6), torch.zeros(2, 6), weights)


@pytest.mark.parametrize(
    "weights",
    [
        torch.tensor([0.0, 1.0]),
        torch.tensor([-1.0, 1.0]),
        torch.tensor([float("nan"), 1.0]),
        torch.ones(3),
    ],
)
def test_invalid_weights_fail_closed(weights):
    with pytest.raises(ValueError, match="ağırlıkları"):
        weighted_video_loss(torch.zeros(1), torch.zeros(1), weights)


@pytest.mark.parametrize(
    "records", [[], [dict(label=0)], [dict(label=1)], [dict(label=2)]]
)
def test_invalid_train_class_counts_rejected(records):
    with pytest.raises(ValueError):
        training_class_balance(records)


def test_sample_is_reproducible_balanced_and_does_not_mutate_split():
    records = train_records()
    before = [r.copy() for r in records]
    selected = select_balanced_sample(records)
    assert len(selected) == 40 and sum(r["label"] == 0 for r in selected) == 20
    assert len({r["path"] for r in selected}) == 40
    assert selected == select_balanced_sample(list(reversed(records)))
    assert selected != select_balanced_sample(records, seed=43)
    assert all(r in records for r in selected)
    assert records == before


@pytest.mark.parametrize("count", [0, 21, 2330])
def test_small_test_cannot_expand_to_full_preprocessing(count):
    with pytest.raises(ValueError):
        select_balanced_sample(train_records(), per_class=count)


def test_small_sample_requires_enough_videos():
    with pytest.raises(ValueError):
        select_balanced_sample(
            [dict(label=0, path="r.mp4"), dict(label=1, path="f.mp4")]
        )


def test_cli_modes_are_exclusive_and_do_not_offer_official_test_cache():
    parser = make_parser()
    assert parser.parse_args(["--sample-cache"]).sample_cache
    with pytest.raises(SystemExit):
        parser.parse_args(["--sample-cache", "--cache-split", "train"])
    with pytest.raises(SystemExit):
        parser.parse_args(["--cache-split", "test"])


def test_eta_respects_full_train_imbalance_and_ignores_cache_hit_speed():
    timings = [
        dict(label=0, elapsed_seconds=1.0, cache_hit=False),
        dict(label=1, elapsed_seconds=3.0, cache_hit=False),
        dict(label=0, elapsed_seconds=0.001, cache_hit=True),
    ]
    estimate = estimate_preprocessing(
        timings, dict(REAL=438, FAKE=1892, total=2330), 5.0
    )
    assert estimate["estimated_seconds"] == pytest.approx(438 + 1892 * 3 + 5)
    assert estimate["fresh_sample_counts"] == {"REAL": 1, "FAKE": 1}
    assert (
        estimate_preprocessing(
            [dict(label=0, elapsed_seconds=0.001, cache_hit=True)],
            dict(REAL=438, FAKE=1892),
            5,
        )
        is None
    )
