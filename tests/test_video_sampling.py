"""Balanced video policy, RNG resume, coverage and unweighted image updates."""

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from src.train import capture_rng, restore_rng
from src.video_finetune import (
    load_balanced_reference,
    make_finetune_loaders,
    make_parser,
    validate_resume,
)
from src.video_pipeline import make_video_loader
from src.video_sampling import (
    gradient_summary,
    make_balanced_video_loader,
    sampling_summary,
    source_metrics,
)
from src.video_train import train_epoch, weighted_video_loss


def roster(real=417, fake=1806):
    return [
        dict(path=f"{source}/{index}.mp4", label=label)
        for label, source, count in (
            (0, "Celeb-real", real),
            (1, "Celeb-synthesis", fake),
        )
        for index in range(count)
    ]


def items(records):
    return [
        dict(
            record=r,
            values=torch.ones(3, 2),
            report=dict(status="accepted", failed_frames=[]),
        )
        for r in records
    ]


def order(loader):
    return [batch["items"][0]["record"] for batch in loader]


def test_balanced_probabilities_budget_and_actual_unique_coverage():
    records = roster()
    loader = make_balanced_video_loader(items(records), records)
    sampler = loader.sampler
    assert sampler.replacement and sampler.num_samples == len(loader) == 2223
    assert loader.batch_size == 1 and sampler.generator is loader.generator
    assert sum(sampler.weights[:417]).item() == pytest.approx(1)
    assert sum(sampler.weights[417:]).item() == pytest.approx(1)
    draws = order(loader)
    assert draws == order(make_balanced_video_loader(items(records), records))
    stats = sampling_summary(records, draws)
    assert stats["total_draws"] == 2223
    for name in ("REAL", "FAKE"):
        values = stats["by_class"][name]
        assert 0.45 < values["sampling_fraction"] < 0.55
        assert (
            values["repeat_draws"] == values["sampled_draws"] - values["unique_videos"]
        )
        assert (
            values["unique_coverage"]
            == values["unique_videos"] / values["available_videos"]
        )
    assert stats["by_class"]["REAL"]["repeated_videos"] > 0
    assert stats["by_class"]["FAKE"]["unique_videos"] < 1806


@pytest.mark.parametrize(
    "invalid",
    [[], [dict(path="a", label=0)], [dict(path="a", label=0), dict(path="a", label=1)]],
)
def test_invalid_or_nonunique_sampler_roster_rejected(invalid):
    with pytest.raises(ValueError):
        make_balanced_video_loader(items(invalid), invalid)


def test_sampling_report_rejects_validation_or_changed_label():
    records = roster(1, 1)
    for draw in (dict(path="val/outside.mp4", label=0), dict(records[0], label=1)):
        with pytest.raises(ValueError):
            sampling_summary(records, [draw])


def test_sampler_rejects_misaligned_dataset_roster():
    class MisalignedDataset:
        records = list(reversed(roster(1, 1)))

        def __len__(self):
            return 2

    with pytest.raises(ValueError, match="sırası"):
        make_balanced_video_loader(MisalignedDataset(), roster(1, 1))


def test_balanced_loader_rng_resumes_next_epoch_exactly():
    records = roster(4, 20)
    train = make_balanced_video_loader(items(records), records)
    val = make_video_loader(items(records), 1)
    order(train)
    saved = capture_rng(train, val)
    expected = order(train)
    restored = make_balanced_video_loader(items(records), records)
    restored_val = make_video_loader(items(records), 1)
    restore_rng(saved, restored, restored_val)
    assert order(restored) == expected
    assert order(val) == order(restored_val) == records


def test_validation_loader_never_balanced_or_shuffled():
    records = roster(2, 10)
    datasets = {name: items(records) for name in ("train", "val")}
    audit = dict(splits={"train": dict(accepted_records=records)})
    loaders = make_finetune_loaders(datasets, audit, True)
    assert order(loaders["val"]) == records
    assert loaders["train"].sampler.replacement


def test_clipping_is_attributed_to_class_not_sample_frequency():
    stats = gradient_summary(
        [
            dict(label=0, pre_clip=3, post_clip=1),
            dict(label=0, pre_clip=0.5, post_clip=0.5),
            dict(label=1, pre_clip=10, post_clip=1),
        ]
    )
    assert stats["REAL"]["clipping_rate"] == 0.5
    assert stats["REAL"]["norm_pre_clip"]["median"] == 1.75
    assert stats["FAKE"]["clipping_rate"] == 1
    assert stats["FAKE"]["norm_post_clip"]["max"] == 1


class SmallVideoModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.head = nn.Linear(2, 1)

    def extract_features(self, values):
        return values

    def forward(self, features, lengths):
        assert torch.all(lengths == 3)
        return self.head(features.mean(1)).squeeze(-1)


def test_unweighted_bce_and_actual_batch_diagnostics(monkeypatch):
    from src import video_train

    records = roster(2, 10)
    loader = make_balanced_video_loader(items(records), records)
    model = SmallVideoModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    original = video_train.weighted_video_loss
    losses = []

    def check(logits, labels, weights):
        assert weights is None
        loss = original(logits, labels, weights)
        torch.testing.assert_close(
            loss, nn.functional.binary_cross_entropy_with_logits(logits, labels)
        )
        losses.append(loss.item())
        return loss

    monkeypatch.setattr(video_train, "weighted_video_loss", check)
    result = train_epoch(
        model,
        loader,
        torch.device("cpu"),
        optimizer,
        torch.amp.GradScaler("cuda", enabled=False),
        True,
        False,
        1,
        class_weights=None,
        track_video_diagnostics=True,
    )
    assert result["optimizer_steps"] == len(losses) == len(records)
    assert result["sampling"]["total_draws"] == len(records)
    assert sum(
        v["optimizer_steps"] for v in result["numerics"]["by_class"].values()
    ) == len(records)
    assert len(result["numerics"]["video_gradient_norms"]) == len(records)
    assert result["numerics"]["max_gradient_norm_post_clip"] <= 1.0001


def test_balanced_optimizer_trajectory_resumes_exactly():
    records = roster(2, 10)
    dataset = items(records)
    loader = make_balanced_video_loader(dataset, records)
    val = make_video_loader(dataset, 1)
    model = SmallVideoModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    device = torch.device("cpu")
    train_epoch(model, loader, device, optimizer, scaler, True, False, 1)
    model_state, optimizer_state = (
        copy.deepcopy(model.state_dict()),
        copy.deepcopy(optimizer.state_dict()),
    )
    rng = capture_rng(loader, val)
    expected = train_epoch(
        model,
        loader,
        device,
        optimizer,
        scaler,
        True,
        False,
        2,
        track_video_diagnostics=True,
    )
    restored = SmallVideoModel()
    restored.load_state_dict(model_state)
    restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=3e-4)
    restored_optimizer.load_state_dict(optimizer_state)
    restored_loader = make_balanced_video_loader(dataset, records)
    restored_val = make_video_loader(dataset, 1)
    restore_rng(rng, restored_loader, restored_val)
    observed = train_epoch(
        restored,
        restored_loader,
        device,
        restored_optimizer,
        scaler,
        True,
        False,
        2,
        track_video_diagnostics=True,
    )
    assert expected["sampling"] == observed["sampling"]
    assert expected["video_loss"] == observed["video_loss"]
    assert expected["numerics"] == observed["numerics"]
    for name, value in model.state_dict().items():
        assert torch.equal(value, restored.state_dict()[name])


def test_unweighted_is_one_bce_per_video():
    logits, labels = torch.tensor([2.0, -1.0]), torch.tensor([0.0, 1.0])
    torch.testing.assert_close(
        weighted_video_loss(logits, labels),
        nn.functional.binary_cross_entropy_with_logits(logits, labels),
    )


def test_source_recall_and_single_class_folders():
    values = [
        ("Celeb-real", 0, 1),
        ("Celeb-real", 0, 0),
        ("YouTube-real", 0, 0),
        ("Celeb-synthesis", 1, 1),
    ]
    rows = [
        dict(
            path=f"{source}/{i}.mp4",
            target=target,
            prediction=pred,
            fake_probability=float(pred),
        )
        for i, (source, target, pred) in enumerate(values)
    ]
    result = source_metrics(rows)
    assert result["Celeb-real"]["recall"] == 0.5
    assert result["YouTube-real"]["recall"] == 1
    assert result["Celeb-synthesis"]["recall"] == 1


def test_balanced_cli_explicit_and_natural_defaults_preserved():
    args = make_parser().parse_args(["--run-training", "--accept-filename-families"])
    assert args.sampling == "natural" and args.reference_run is None
    args = make_parser().parse_args(
        [
            "--balanced-smoke",
            "--accept-filename-families",
            "--sampling",
            "balanced",
            "--fresh-head",
            "--epochs",
            "3",
        ]
    )
    assert args.balanced_smoke and not args.run_training


@pytest.mark.parametrize(
    "overrides", [dict(fresh_head=False), dict(epochs=4), dict(reference_run=None)]
)
def test_balanced_requires_fresh_same_budget_and_reference(overrides):
    args = SimpleNamespace(fresh_head=True, epochs=3, reference_run=Path("missing"))
    args.__dict__.update(overrides)
    with pytest.raises(ValueError, match="Balanced karşılaştırma"):
        load_balanced_reference(
            args,
            dict(schema="celeb_df_identity_components_v1"),
            dict(training_class_balance=dict(counts={"REAL": 417, "FAKE": 1806})),
        )


def test_resume_rejects_balancing_policy_change():
    config = dict(sampling="balanced", loss="unweighted_video_bce")
    saved = dict(
        pipeline="video_partial_finetune_v1",
        class_names=["REAL", "FAKE"],
        manifest_fingerprint="f",
        config=config,
    )
    for key in (
        "model_state_dict",
        "optimizer_state_dict",
        "scaler_state_dict",
        "rng_state",
        "early_stopping",
        "history",
        "epoch",
        "best_epoch",
        "best_macro_f1",
    ):
        saved[key] = {}
    validate_resume(saved, dict(fingerprint="f"), config)
    altered = copy.deepcopy(config)
    altered["loss"] = "weighted_bce"
    with pytest.raises(ValueError):
        validate_resume(saved, dict(fingerprint="f"), altered)


def test_reference_setting_mismatch_rejected_before_checkpoint_read(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps(dict(head_lr=3e-4)))
    args = SimpleNamespace(fresh_head=True, epochs=3, reference_run=tmp_path)
    config = dict(
        head_lr=1e-2, training_class_balance=dict(counts={"REAL": 417, "FAKE": 1806})
    )
    with pytest.raises(ValueError, match="head_lr"):
        load_balanced_reference(
            args, dict(schema="celeb_df_identity_components_v1"), config
        )


def test_balanced_training_wiring_budget_and_fixed_best(tmp_path, monkeypatch):
    from src import video_finetune, video_train
    from src.video_evaluate import evaluate_video
    from src.video_sampling import comparison_metrics
    from src.video_train import training_class_balance

    monkeypatch.chdir(tmp_path)
    torch.set_num_threads(2)
    records = roster()
    val_records = [records[0], dict(path="YouTube-real/0.mp4", label=0), records[-1]]
    datasets = dict(train=items(records), val=items(val_records))
    audit = dict(
        splits={
            name: dict(
                total=len(rows),
                accepted=len(rows),
                rejected=0,
                accepted_counts={
                    "REAL": sum(r["label"] == 0 for r in rows),
                    "FAKE": sum(r["label"] == 1 for r in rows),
                },
                accepted_records=rows,
                rejected_reports=[],
            )
            for name, rows in (("train", records), ("val", val_records))
        }
    )
    model = SmallVideoModel()
    manifest = dict(fingerprint="synthetic", dataset_root=str(tmp_path))
    monkeypatch.setattr(
        video_finetune,
        "prepare_context",
        lambda args: (
            manifest,
            model,
            None,
            audit,
            training_class_balance(records),
            dict(seed=42),
            datasets,
        ),
    )
    reference_eval = evaluate_video(
        model, make_video_loader(datasets["val"], 1), torch.device("cpu")
    )
    reference = dict(validation=comparison_metrics(reference_eval))
    monkeypatch.setattr(
        video_finetune, "load_balanced_reference", lambda *args: reference
    )
    monkeypatch.setattr(
        video_finetune,
        "make_optimizer",
        lambda model, *args: torch.optim.AdamW(model.parameters(), lr=3e-4),
    )
    original_loss = video_train.weighted_video_loss
    observed_weights = []

    def checked_loss(logits, labels, weights):
        observed_weights.append(weights)
        assert weights is None
        return original_loss(logits, labels, weights)

    monkeypatch.setattr(video_train, "weighted_video_loss", checked_loss)
    args = make_parser().parse_args(
        [
            "--run-training",
            "--sampling",
            "balanced",
            "--fresh-head",
            "--accept-filename-families",
            "--device",
            "cpu",
            "--epochs",
            "1",
        ]
    )
    video_finetune.run_training(args)
    run_dir = args.run_dir
    checkpoint = torch.load(run_dir / "checkpoints/best_video.pth", weights_only=True)
    assert len(observed_weights) == checkpoint["history"][0]["optimizer_steps"] == 2223
    assert checkpoint["config"]["loss_class_weights"] is None
    assert checkpoint["config"]["loss"] == "unweighted_video_bce"
    assert checkpoint["rng_state"]["train_generator"] is not None
    fixed = json.loads((run_dir / "best_fixed_evaluation.json").read_text())
    assert fixed["epoch"] == 1
    assert len(fixed["results"]["train"]["predictions"]) == 2223
    assert len(fixed["results"]["val"]["predictions"]) == 3
    assert fixed["scope"] == "fixed_best_checkpoint_eval_no_optimizer_steps"
    # Completed resume repeats neither optimizer steps nor fixed evaluation.
    args.resume = "auto"
    video_finetune.run_training(args)
    assert len(observed_weights) == 2223
    assert not (run_dir / "checkpoints/epoch_002.pth").exists()
