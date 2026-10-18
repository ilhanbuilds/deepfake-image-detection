"""Sentetik fotoğraflarla V3 augmentation ve güvenlik; eğitim başlatılmaz."""

import copy
import io
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from torchvision.transforms import functional as TF

import src.train_v3 as train_v3
from src.dataset import image_transform, make_loader
from src.presplit_dataset import dataset_fingerprint, split_counts
from src.robust_augmentation import (
    AugmentationSettings,
    RobustPresplitDataset,
    RobustTrainTransform,
    image_transform_v3,
)
from src.train import capture_rng, restore_rng


def photo():
    rng = np.random.default_rng(4)
    return Image.fromarray(rng.integers(0, 256, (256, 256, 3), dtype=np.uint8))


def disabled_settings(**changes):
    return replace(
        AugmentationSettings(),
        **{
            "flip_probability": 0,
            "jpeg_probability": 0,
            "blur_probability": 0,
            "noise_probability": 0,
            "resize_probability": 0,
            **changes,
        },
    )


@pytest.fixture
def root(tmp_path):
    for split in ("train", "valid"):
        for folder in ("real", "fake"):
            (tmp_path / split / folder).mkdir(parents=True)
            photo().save(tmp_path / split / folder / "a.png")
    return tmp_path


def test_disabled_augmentations_and_eval_equal_clean_exactly():
    image = photo()
    expected = image_transform(False)(image)
    torch.testing.assert_close(
        RobustTrainTransform(disabled_settings())(image), expected, rtol=0, atol=0
    )
    state = torch.get_rng_state().clone()
    # Eval factory ignores augmentation settings and does not consume RNG.
    forced = replace(AugmentationSettings(), noise_probability=1, blur_probability=1)
    for _ in range(3):
        torch.testing.assert_close(
            image_transform_v3(False, forced)(image), expected, rtol=0, atol=0
        )
    assert torch.equal(state, torch.get_rng_state())


@pytest.mark.parametrize("kind", ["flip", "jpeg", "blur", "noise", "resize"])
def test_each_forced_augmentation_changes_photo_without_mutating_original(kind):
    image = photo()
    original = image.tobytes()
    settings = disabled_settings(**{f"{kind}_probability": 1})
    transformed = RobustTrainTransform(settings)(image)
    assert transformed.shape == (3, 224, 224) and torch.isfinite(transformed).all()
    assert image.tobytes() == original
    assert not torch.equal(transformed, image_transform(False)(image))
    rgb = TF.normalize(
        transformed, mean=(0, 0, 0), std=(1 / 0.229, 1 / 0.224, 1 / 0.225)
    )
    rgb += torch.tensor([0.485, 0.456, 0.406])[:, None, None]
    assert rgb.min() >= -1e-6 and rgb.max() <= 1 + 1e-6


@pytest.mark.parametrize("quality", [50, 95])
def test_jpeg_quality_endpoints_roundtrip(quality):
    image = photo()
    settings = disabled_settings(jpeg_probability=1, jpeg_quality=(quality, quality))
    with io.BytesIO() as buffer:
        TF.resize(image, [224, 224], antialias=True).save(
            buffer, format="JPEG", quality=quality, subsampling=2
        )
        buffer.seek(0)
        with Image.open(buffer) as decoded:
            expected = image_transform(False)(decoded.convert("RGB"))
    torch.testing.assert_close(
        RobustTrainTransform(settings)(image), expected, rtol=0, atol=0
    )


def test_noise_zero_std_and_resize_224_are_identity():
    image = photo()
    transform = RobustTrainTransform(
        disabled_settings(
            noise_probability=1,
            noise_std=(0, 0),
            resize_probability=1,
            resize_size=(224, 224),
        )
    )
    torch.testing.assert_close(
        transform(image), image_transform(False)(image), rtol=0, atol=0
    )


def test_sampling_ranges_and_probability_frequencies():
    torch.manual_seed(42)
    transform = RobustTrainTransform()
    for bounds in ((50, 95), (112, 224)):
        values = [transform._integer(bounds) for _ in range(500)]
        assert min(values) == bounds[0] and max(values) == bounds[1]
    for bounds in ((0.1, 1.5), (0, 0.02)):
        values = [transform._uniform(bounds) for _ in range(500)]
        assert all(bounds[0] <= value <= bounds[1] for value in values)
        assert len(set(values)) > 490
    for probability in (0.15, 0.30, 0.50):
        observed = sum(transform._apply(probability) for _ in range(4000)) / 4000
        assert abs(observed - probability) < 0.03


@pytest.mark.parametrize(
    "changes",
    [
        {"blur_probability": -0.1},
        {"noise_probability": float("nan")},
        {"jpeg_quality": (95, 50)},
        {"jpeg_quality": (50, 101)},
        {"resize_size": (112, 225)},
        {"blur_sigma": (0, 1.5)},
        {"noise_std": (0, float("inf"))},
    ],
)
def test_invalid_augmentation_settings_rejected(changes):
    with pytest.raises(ValueError):
        RobustTrainTransform(replace(AugmentationSettings(), **changes))


def test_dataset_mapping_membership_and_train_only_augmentation(root, monkeypatch):
    (root / "test").mkdir()
    original_iterdir = Path.iterdir

    def guarded(path):
        if path == root / "test":
            raise AssertionError("Training test klasörüne erişti.")
        return original_iterdir(path)

    monkeypatch.setattr(Path, "iterdir", guarded)
    datasets = {
        split: RobustPresplitDataset(root, split) for split in ("train", "valid")
    }
    assert split_counts(datasets) == {
        "train": {"REAL": 1, "FAKE": 1, "total": 2},
        "valid": {"REAL": 1, "FAKE": 1, "total": 2},
    }
    assert isinstance(datasets["train"].transform, RobustTrainTransform)
    assert not isinstance(datasets["valid"].transform, RobustTrainTransform)
    assert not set(datasets["train"].samples) & set(datasets["valid"].samples)
    for dataset in datasets.values():
        assert dataset.class_to_idx == {"real": 0, "fake": 1}
        assert dataset.targets == [label for _, label in dataset.samples]
        images, labels = next(iter(make_loader(dataset, batch_size=2)))
        assert images.shape == (2, 3, 224, 224) and sorted(labels.tolist()) == [0, 1]
    torch.testing.assert_close(
        datasets["valid"][0][0], datasets["valid"][0][0], rtol=0, atol=0
    )
    assert len(dataset_fingerprint(datasets)) == 64
    with pytest.raises(ValueError):
        RobustPresplitDataset(root, "test")


def test_checkpoint_rng_reproduces_shuffled_augmented_next_epoch(root):
    datasets = {
        split: RobustPresplitDataset(root, split) for split in ("train", "valid")
    }
    loaders = {
        name: make_loader(ds, batch_size=2, shuffle=name == "train", seed=42)
        for name, ds in datasets.items()
    }
    torch.manual_seed(42)
    state = capture_rng(loaders["train"], loaders["valid"])
    expected_images, expected_labels = next(iter(loaders["train"]))
    torch.rand(20)
    restore_rng(state, loaders["train"], loaders["valid"])
    actual_images, actual_labels = next(iter(loaders["train"]))
    torch.testing.assert_close(actual_images, expected_images, rtol=0, atol=0)
    assert torch.equal(actual_labels, expected_labels)


def test_v3_cli_defaults_and_explicit_start():
    args = train_v3.make_parser().parse_args(["--run-training"])
    assert (args.epochs, args.batch_size, args.patience, args.seed, args.workers) == (
        6,
        8,
        2,
        42,
        0,
    )
    assert args.learning_rate == 3e-4 and args.device == "cuda"
    assert (
        train_v3.make_parser()
        .parse_args(["--run-training", "--epochs", "1", "--resume", "auto"])
        .resume
        == "auto"
    )
    with pytest.raises(SystemExit):
        train_v3.make_parser().parse_args([])


def test_v3_resume_rejects_v2_wrong_labels_changed_augmentation(root):
    config = {"augmentation": asdict(AugmentationSettings())}
    checkpoint = {
        "pipeline": "presplit_image_v3_robust",
        "dataset_fingerprint": "fp",
        "config": config,
        "dataset_root": str(root),
        "class_names": ["REAL", "FAKE"],
        "image_size": 224,
        **{
            key: {}
            for key in (
                "optimizer_state_dict",
                "scaler_state_dict",
                "rng_state",
                "early_stopping",
                "history",
                "best_epoch",
            )
        },
    }
    train_v3.validate_v3_resume(checkpoint, "fp", config, root)
    for key, value in (
        ("pipeline", "presplit_image_v2"),
        ("class_names", ["FAKE", "REAL"]),
        ("dataset_fingerprint", "changed"),
        ("config", {"augmentation": {}}),
    ):
        invalid = copy.deepcopy(checkpoint)
        invalid[key] = value
        with pytest.raises(ValueError):
            train_v3.validate_v3_resume(invalid, "fp", config, root)


def test_existing_v3_checkpoint_and_v2_resume_fail_before_model_loading(
    root, monkeypatch
):
    project = root / "project"
    protected = project / "outputs/checkpoints/best_model_v2.pth"
    protected.parent.mkdir(parents=True)
    protected.write_bytes(b"unchanged V2")
    monkeypatch.setattr(train_v3, "PROJECT_ROOT", project)
    monkeypatch.setattr(train_v3, "resolve_device", lambda _: torch.device("cpu"))

    def forbidden(*args, **kwargs):
        raise AssertionError("Model/eğitim başlatılmamalı.")

    monkeypatch.setattr(train_v3, "build_model", forbidden)
    args = train_v3.make_parser().parse_args(
        ["--run-training", "--data-dir", str(root)]
    )
    v3 = project / "outputs/checkpoints/best_model_v3_robust.pth"
    v3.write_bytes(b"existing V3")
    with pytest.raises(FileExistsError):
        train_v3.run_training(args, {})
    assert v3.read_bytes() == b"existing V3"
    v3.unlink()
    args.resume = str(protected)
    with pytest.raises(ValueError, match="image_v3_robust"):
        train_v3.run_training(args, {})
    assert protected.read_bytes() == b"unchanged V2"
    assert not (project / "outputs/checkpoints/image_v3_robust").exists()
