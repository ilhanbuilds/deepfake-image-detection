"""Actual image gradients, frozen BN, temporal masks and separate resume safety."""

import copy
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from src.train import EarlyStopping, capture_rng, restore_rng
from src.video_evaluate import load_video_checkpoint
from src.video_finetune import make_parser, run_training, validate_resume
from src.video_finetune_data import CropDriftError, VerifiedImageDataset
from src.video_finetune_model import PIPELINE, PartialVideoClassifier, make_optimizer
from src.video_pipeline import make_video_loader, video_collate
from src.video_train import (
    train_video_batch,
    training_class_balance,
    weighted_video_loss,
)


@pytest.fixture
def model():
    torch.set_num_threads(2)
    torch.manual_seed(42)
    return PartialVideoClassifier(pretrained=False)


def image_item(label=0, length=6):
    return dict(
        record=dict(path=f"train/{label}.mp4", label=label),
        values=torch.randn(length, 3, 160, 160),
        report=dict(status="accepted", valid_frames=length),
    )


def test_only_final_feature_block_gru_and_head_trainable(model):
    names = {n for n, p in model.named_parameters() if p.requires_grad}
    assert len(model.backbone[0]) == 9
    assert names == {
        n
        for n, _ in model.named_parameters()
        if n.startswith(("backbone.0.8.", "gru.", "head."))
    }
    assert not model.backbone[0][7].training
    groups = make_optimizer(model).param_groups
    assert [g["lr"] for g in groups] == [1e-5, 3e-4]
    assert {id(p) for g in groups for p in g["params"]} == {
        id(p) for p in model.parameters() if p.requires_grad
    }


@pytest.mark.parametrize("length", [3, 4, 5, 6])
def test_real_image_update_and_bn_running_stats_fixed(model, length):
    before = {n: v.clone() for n, v in model.state_dict().items()}
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    result = train_video_batch(
        model,
        video_collate([image_item(0, length)]),
        torch.device("cpu"),
        make_optimizer(model),
        scaler,
        False,
        False,
        torch.tensor([2.68, 0.61]),
    )
    assert result["finite"] and result["lengths"] == [length]
    changed = {
        n for n, v in model.state_dict().items() if not torch.equal(v, before[n])
    }
    assert all(n.startswith(("backbone.0.8.", "gru.", "head.")) for n in changed)
    assert all(
        any(n.startswith(prefix) for n in changed)
        for prefix in ("backbone.0.8.0.", "backbone.0.8.1.", "gru.", "head.")
    )
    assert not any(
        n.endswith(("running_mean", "running_var", "num_batches_tracked"))
        for n in changed
    )
    assert all(p.grad is None for p in model.backbone[0][:-1].parameters())
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all()
        for p in model.backbone[0][-1].parameters()
    )
    for mode in (False, True, True):
        model.train(mode)
        assert all(
            not m.training
            for m in model.backbone.modules()
            if isinstance(m, nn.BatchNorm2d)
        )


def test_mixed_image_sequences_nan_padding_ignored(model):
    model.eval()
    with torch.no_grad():
        features = model.extract_features(torch.randn(6, 3, 160, 160))
        padded = features.repeat(4, 1, 1)
        lengths = torch.tensor([3, 4, 5, 6])
        for index, length in enumerate(lengths):
            padded[index, length:] = float("nan")
        observed = model(padded, lengths)
        expected = torch.cat(
            [model(features[None, :length], length[None]) for length in lengths]
        )
    torch.testing.assert_close(observed, expected, atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize(
    "shape", [(6, 1280), (2, 3, 160, 160), (7, 3, 160, 160), (6, 3, 192, 192)]
)
def test_feature_cache_and_invalid_image_inputs_rejected(model, shape):
    with pytest.raises(ValueError, match="görüntü"):
        model.extract_features(torch.zeros(shape))


def test_weighted_bce_is_single_video_loss_not_frame_loss():
    balance = training_class_balance(
        [dict(label=0), *[dict(label=1) for _ in range(3)]]
    )
    assert balance["class_weights"] == {"REAL": 2.0, "FAKE": 2 / 3}
    logits, labels = torch.tensor([0.2, -0.3]), torch.tensor([0.0, 1.0])
    expected = nn.functional.binary_cross_entropy_with_logits(
        logits, labels, weight=torch.tensor([2.0, 2 / 3])
    )
    torch.testing.assert_close(
        weighted_video_loss(logits, labels, torch.tensor([2.0, 2 / 3])), expected
    )


@pytest.mark.parametrize(
    "key,value",
    [
        ("valid_indices", [0, 2, 4]),
        ("status", "rejected"),
        ("boxes", [[5, 5, 5, 5]]),
        ("failed_frames", [{"index": 5, "reason": "face_not_detected"}]),
    ],
)
def test_fresh_crop_drift_fails_instead_of_silent_rejection(key, value):
    dataset = object.__new__(VerifiedImageDataset)
    expected = dict(
        status="accepted",
        valid_indices=list(range(6)),
        selected_indices=list(range(6)),
        boxes=[],
        failed_frames=[],
    )
    dataset.expected = {"a": expected}
    frames = torch.zeros(6, 3, 160, 160)
    dataset.verify(dict(path="a"), frames, expected)
    with pytest.raises(CropDriftError):
        dataset.verify(dict(path="a"), frames, dict(expected, **{key: value}))
    with pytest.raises(CropDriftError):
        dataset.verify(dict(path="a"), None, expected)


def test_resume_retains_image_optimizer_rng_and_stopper(model):
    items = [image_item(0, 3), image_item(1, 6)]
    loader = make_video_loader(items, 1, shuffle=True)
    val_loader = make_video_loader(items, 1)
    optimizer = make_optimizer(model)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    for batch in loader:
        train_video_batch(
            model, batch, torch.device("cpu"), optimizer, scaler, False, False
        )
    stopper = EarlyStopping(patience=3)
    stopper.update(0.6)
    stopper.update(0.5)
    rng = capture_rng(loader, val_loader)
    state, optim_state = (
        copy.deepcopy(model.state_dict()),
        copy.deepcopy(optimizer.state_dict()),
    )
    expected_order = []
    for batch in loader:
        expected_order.extend(i["record"]["path"] for i in batch["items"])
        train_video_batch(
            model, batch, torch.device("cpu"), optimizer, scaler, False, False
        )
    expected = copy.deepcopy(model.state_dict())
    resumed = PartialVideoClassifier(pretrained=False)
    resumed.load_state_dict(state)
    restored_optimizer = make_optimizer(resumed)
    restored_optimizer.load_state_dict(optim_state)
    resumed_loader, resumed_val = (
        make_video_loader(items, 1, shuffle=True),
        make_video_loader(items, 1),
    )
    restore_rng(rng, resumed_loader, resumed_val)
    actual_order = []
    for batch in resumed_loader:
        actual_order.extend(i["record"]["path"] for i in batch["items"])
        train_video_batch(
            resumed,
            batch,
            torch.device("cpu"),
            restored_optimizer,
            scaler,
            False,
            False,
        )
    assert actual_order == expected_order
    assert asdict(EarlyStopping(**asdict(stopper))) == asdict(stopper)
    for name, value in resumed.state_dict().items():
        assert torch.equal(value, expected[name]), name


def test_partial_checkpoint_is_not_accepted_by_frozen_loader(tmp_path):
    checkpoint = tmp_path / "partial.pth"
    torch.save(dict(pipeline=PIPELINE), checkpoint)
    with pytest.raises(ValueError):
        load_video_checkpoint(checkpoint, torch.device("cpu"))


def test_partial_resume_refuses_frozen_or_missing_state():
    with pytest.raises(ValueError, match="resume"):
        validate_resume(dict(pipeline="video_v1_frozen_gru"), dict(fingerprint="f"), {})
    with pytest.raises(ValueError, match="resume"):
        validate_resume(
            dict(
                pipeline=PIPELINE,
                class_names=["REAL", "FAKE"],
                manifest_fingerprint="f",
                config={},
            ),
            dict(fingerprint="f"),
            {},
        )


def test_training_cli_is_explicit_and_fixed_safe_defaults():
    args = make_parser().parse_args(["--run-training", "--accept-filename-families"])
    assert args.epochs == 1 and args.backbone_lr == 1e-5 and args.head_lr == 3e-4
    with pytest.raises(SystemExit):
        make_parser().parse_args([])


def test_existing_video_output_directory_is_refused_before_loading():
    with pytest.raises(ValueError, match="ayrı"):
        run_training(
            SimpleNamespace(run_dir=Path("outputs/video_v1/gru_v2_hidden64_fp32"))
        )


def synthetic_context(model, tmp_path):
    items = [image_item(0, 3), image_item(1, 6)]
    for item in items:
        item["report"].update(failed_frames=[], relative_path=item["record"]["path"])
    records = [i["record"] for i in items]
    audit = dict(
        splits={
            name: dict(
                total=2,
                accepted=2,
                rejected=0,
                accepted_counts={"REAL": 1, "FAKE": 1},
                accepted_records=records,
                rejected_reports=[],
            )
            for name in ("train", "val")
        }
    )
    return (
        dict(fingerprint="synthetic", dataset_root=str(tmp_path)),
        model,
        None,
        audit,
        training_class_balance(records),
        {"seed": 42},
        {"train": items, "val": items},
    )


def test_training_checkpoint_resume_and_validation_only_selection(
    model, tmp_path, monkeypatch
):
    from src import video_finetune as module

    monkeypatch.chdir(tmp_path)
    context = synthetic_context(model, tmp_path)
    monkeypatch.setattr(module, "prepare_context", lambda args: context)
    actual_evaluate = module.evaluate_video
    scores = iter([0.7, 0.6, 0.5, 0.4])

    def controlled_validation(*args, **kwargs):
        result = actual_evaluate(*args, **kwargs)
        result["metrics"]["macro_f1"] = next(scores)
        return result

    monkeypatch.setattr(module, "evaluate_video", controlled_validation)
    args = make_parser().parse_args(
        ["--run-training", "--accept-filename-families", "--device", "cpu"]
    )
    run_training(args)
    directory = args.run_dir / "checkpoints"
    checkpoint = torch.load(directory / "epoch_001.pth", weights_only=True)
    assert checkpoint["pipeline"] == PIPELINE
    assert checkpoint["best_macro_f1"] == 0.7
    assert checkpoint["rng_state"] and checkpoint["optimizer_state_dict"]["state"]
    args.epochs, args.resume = 10, "auto"
    run_training(args)
    last = torch.load(directory / "last_video.pth", weights_only=True)
    best = torch.load(directory / "best_video.pth", weights_only=True)
    assert last["epoch"] == 4 and last["early_stopping"]["bad_epochs"] == 3
    assert best["epoch"] == 1 and last["best_epoch"] == 1
    assert len(last["history"]) == 4
    assert set(last["history"][-1]["validation_metrics"]) >= {
        "macro_f1",
        "balanced_accuracy",
        "roc_auc",
        "per_class",
    }
    run_training(args)
    assert not (directory / "epoch_005.pth").exists()


@pytest.mark.parametrize("failure", [torch.cuda.OutOfMemoryError, CropDriftError])
def test_partial_training_failure_records_recovery_without_checkpoint(
    model, tmp_path, monkeypatch, failure
):
    import json
    from src import video_finetune as module

    monkeypatch.chdir(tmp_path)
    context = synthetic_context(model, tmp_path)
    monkeypatch.setattr(module, "prepare_context", lambda args: context)

    def fail(*args, **kwargs):
        raise failure("synthetic failure")

    monkeypatch.setattr(module, "train_epoch", fail)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    args = make_parser().parse_args(
        ["--run-training", "--accept-filename-families", "--device", "cpu"]
    )
    with pytest.raises(failure):
        run_training(args)
    recovery = json.loads(
        (args.run_dir / "interrupted.json").read_text(encoding="utf-8")
    )
    assert (
        recovery["last_completed_epoch"] == 0 and recovery["reason"] == failure.__name__
    )
    assert not list((args.run_dir / "checkpoints").glob("*.pth"))
