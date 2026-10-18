"""Spatial invariance, valid-frame gradients, frozen BN and safe video training."""

import copy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from src.video_finetune import validate_resume as validate_gru_resume
from src.video_pipeline import video_collate
from src.video_spatial import make_parser, run, validate_resume
from src.video_spatial_evaluate import (
    spatial_source_metrics,
    validate_recorded_predictions,
)
from src.video_spatial_model import (
    PIPELINE,
    SpatialVideoClassifier,
    make_spatial_optimizer,
    masked_feature_mean,
)
from src.video_train import train_video_batch


@pytest.mark.parametrize("length", [3, 4, 5, 6])
def test_masked_mean_ignores_nan_padding_and_its_gradient(length):
    torch.manual_seed(42)
    raw = torch.randn(1, 6, 1280)
    raw[:, length:] = float("nan")
    features = raw.requires_grad_()
    actual = masked_feature_mean(features, torch.tensor([length]))
    torch.testing.assert_close(actual, raw[:, :length].mean(1))
    actual.sum().backward()
    torch.testing.assert_close(
        features.grad[:, :length],
        torch.full_like(features.grad[:, :length], 1 / length),
    )
    assert torch.count_nonzero(features.grad[:, length:]) == 0
    assert torch.isfinite(features.grad).all()


def test_permutation_invariance_and_mixed_lengths():
    model = SpatialVideoClassifier(pretrained=False).eval()
    features = torch.randn(4, 6, 1280)
    lengths = torch.tensor([3, 4, 5, 6])
    reversed_values = features.clone()
    for index, length in enumerate(lengths):
        reversed_values[index, :length] = features[index, :length].flip(0)
        features[index, length:] = float("nan")
        reversed_values[index, length:] = float("nan")
    torch.testing.assert_close(
        model(features, lengths), model(reversed_values, lengths)
    )
    assert not any(isinstance(m, nn.GRU) for m in model.modules())
    assert not any("gru" in name for name in model.state_dict())
    assert model(features, lengths).shape == (4,)


@pytest.mark.parametrize(
    "lengths",
    [torch.tensor([2]), torch.tensor([7]), torch.tensor([3.5]), torch.tensor([3, 4])],
)
def test_invalid_lengths_rejected(lengths):
    with pytest.raises(ValueError):
        masked_feature_mean(torch.zeros(1, 6, 1280), lengths)


def test_invalid_valid_features_rejected():
    features = torch.zeros(1, 6, 1280)
    features[0, 0, 0] = float("inf")
    with pytest.raises(ValueError, match="NaN/Inf"):
        masked_feature_mean(features, torch.tensor([3]))


@pytest.mark.parametrize("mode", ["partial", "frozen"])
def test_trainable_parameters_optimizer_groups_and_bn(mode):
    model = SpatialVideoClassifier(pretrained=False, backbone_mode=mode)
    names = {name for name, p in model.named_parameters() if p.requires_grad}
    allowed = ("head.", "backbone.0.8.") if mode == "partial" else ("head.",)
    assert names == {
        name for name, _ in model.named_parameters() if name.startswith(allowed)
    }
    optimizer = make_spatial_optimizer(model)
    assert [g["lr"] for g in optimizer.param_groups] == (
        [1e-5, 3e-4] if mode == "partial" else [3e-4]
    )
    assert all(g["weight_decay"] == 0.01 for g in optimizer.param_groups)
    assert {id(p) for g in optimizer.param_groups for p in g["params"]} == {
        id(p) for p in model.parameters() if p.requires_grad
    }
    for flag in (True, False, True):
        model.train(flag)
        assert not model.backbone[0][7].training
        assert all(
            not m.training
            for m in model.backbone.modules()
            if isinstance(m, nn.BatchNorm2d)
        )


@pytest.mark.parametrize("mode", ["partial", "frozen"])
def test_real_image_single_weighted_video_update(mode):
    torch.set_num_threads(2)
    torch.manual_seed(42)
    model = SpatialVideoClassifier(pretrained=False, backbone_mode=mode)
    before = copy.deepcopy(model.state_dict())
    item = dict(
        record=dict(path="Celeb-real/unit.mp4", label=0),
        values=torch.randn(6, 3, 160, 160),
        report=dict(status="accepted", failed_frames=[]),
    )
    result = train_video_batch(
        model,
        video_collate([item]),
        torch.device("cpu"),
        make_spatial_optimizer(model),
        torch.amp.GradScaler("cuda", enabled=False),
        False,
        False,
        torch.tensor([2.6654676, 0.6154485]),
    )
    changed = {
        name
        for name, value in model.state_dict().items()
        if not torch.equal(value, before[name])
    }
    assert any(name.startswith("head.") for name in changed)
    assert result["logits"].shape == (1,) and result["finite"]
    assert result["gradient_norm_post_clip"] <= 1.0001
    assert not any(
        name.endswith(("running_mean", "running_var", "num_batches_tracked"))
        for name in changed
    )
    if mode == "partial":
        assert any(name.startswith("backbone.0.8.") for name in changed)
        assert all(name.startswith(("head.", "backbone.0.8.")) for name in changed)
    else:
        assert all(name.startswith("head.") for name in changed)


def test_source_pair_auc_and_per_source_recall():
    rows = [
        dict(
            path=source + "/a.mp4",
            target=label,
            fake_probability=score,
            prediction=int(score >= 0.5),
        )
        for source, label, score in (
            ("Celeb-real", 0, 0.8),
            ("YouTube-real", 0, 0.1),
            ("Celeb-synthesis", 1, 0.7),
        )
    ]
    result = spatial_source_metrics(rows)
    assert result["Celeb-real"]["recall"] == 0
    assert result["Celeb-real"]["roc_auc_vs_celeb_synthesis"] == 0
    assert result["YouTube-real"]["roc_auc_vs_celeb_synthesis"] == 1
    assert result["Celeb-synthesis"]["single_source_roc_auc"] is None


def test_recorded_evaluation_rejects_wrong_threshold_or_roster():
    row = dict(path="Celeb-real/a.mp4", target=0, prediction=0, fake_probability=0.8)
    with pytest.raises(ValueError, match="threshold"):
        validate_recorded_predictions(
            dict(predictions=[row]), [dict(path=row["path"], label=0)]
        )
    with pytest.raises(ValueError, match="roster"):
        validate_recorded_predictions(
            dict(predictions=[row]), [dict(path="other", label=0)]
        )


def test_spatial_resume_never_accepts_gru_checkpoint():
    with pytest.raises(ValueError):
        validate_resume(
            dict(pipeline="video_partial_finetune_v1"), dict(fingerprint="f"), {}
        )
    with pytest.raises(ValueError):
        validate_gru_resume(dict(pipeline=PIPELINE), dict(fingerprint="f"), {})


def test_training_cli_and_old_output_protection():
    args = make_parser().parse_args(
        [
            "--run-training",
            "--accept-filename-families",
            "--run-dir",
            "outputs/video_spatial_v1/new",
        ]
    )
    assert (
        args.epochs == 3 and args.backbone_mode == "partial" and args.device == "cuda"
    )
    with pytest.raises(SystemExit):
        make_parser().parse_args(["--run-dir", "x"])
    with pytest.raises(ValueError, match="Ayrı"):
        run(SimpleNamespace(epochs=3, run_dir=Path("outputs/video_finetune_v1/old")))
    with pytest.raises(ValueError, match="3 epoch"):
        run(SimpleNamespace(epochs=4))


def test_spatial_driver_weighted_budget_checkpoint_and_resume(tmp_path, monkeypatch):
    import json
    from src import video_spatial, video_train
    from src.video_train import training_class_balance

    monkeypatch.chdir(tmp_path)
    torch.set_num_threads(2)
    records = [
        dict(path=f"{source}/{i}.mp4", label=label)
        for source, label, count in (
            ("Celeb-real", 0, 415),
            ("YouTube-real", 0, 2),
            ("Celeb-synthesis", 1, 1806),
        )
        for i in range(count)
    ]
    vals = [records[0], records[415], records[-1]]
    values = {0: torch.ones(3, 1280), 1: -torch.ones(3, 1280)}
    datasets = {
        name: [
            dict(
                record=r,
                values=values[r["label"]],
                report=dict(status="accepted", failed_frames=[]),
            )
            for r in rows
        ]
        for name, rows in (("train", records), ("val", vals))
    }
    audit = dict(
        splits={
            name: dict(accepted_records=rows, rejected_reports=[])
            for name, rows in (("train", records), ("val", vals))
        }
    )
    balance = training_class_balance(records)
    config = dict(
        feature_cache_used_as_input=True, training_class_balance=balance, reference={}
    )
    model = SpatialVideoClassifier(pretrained=False, backbone_mode="frozen")
    manifest = dict(fingerprint="synthetic")
    monkeypatch.setattr(
        video_spatial,
        "prepare_context",
        lambda args: (manifest, model, audit, copy.deepcopy(config), datasets),
    )
    original = video_train.weighted_video_loss
    calls = []

    def verify_weighted(logits, labels, class_weights):
        torch.testing.assert_close(
            class_weights, torch.tensor([2223 / (2 * 417), 2223 / (2 * 1806)])
        )
        calls.append(1)
        return original(logits, labels, class_weights)

    monkeypatch.setattr(video_train, "weighted_video_loss", verify_weighted)
    args = make_parser().parse_args(
        [
            "--run-training",
            "--accept-filename-families",
            "--device",
            "cpu",
            "--backbone-mode",
            "frozen",
            "--epochs",
            "1",
            "--run-dir",
            "outputs/video_spatial_v1/unit",
        ]
    )
    run(args)
    assert len(calls) == 2223
    saved = torch.load(args.run_dir / "checkpoints/epoch_001.pth", weights_only=True)
    assert saved["pipeline"] == PIPELINE and saved["rng_state"]
    assert (
        saved["optimizer_state_dict"]["state"]
        and saved["early_stopping"]["patience"] == 3
    )
    assert saved["history"][0]["optimizer_steps"] == 2223
    assert (args.run_dir / "best_fixed_epoch_001_after_001.json").exists()
    args.resume, args.epochs = "auto", 2
    run(args)
    assert len(calls) == 4446
    last = torch.load(args.run_dir / "checkpoints/last_video.pth", weights_only=True)
    assert last["epoch"] == 2 and len(last["history"]) == 2
    fixed = json.loads((args.run_dir / "best_fixed_evaluation.json").read_text())
    assert (
        fixed["epoch"] == last["best_epoch"] and fixed["completed_training_epochs"] == 2
    )
    assert len(fixed["results"]["train"]["predictions"]) == 2223
    run(args)
    assert len(calls) == 4446
    assert not (args.run_dir / "checkpoints/epoch_003.pth").exists()
