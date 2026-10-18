"""Hazır ayrımlar, sınıf eşlemesi ve v2 resume güvenliği."""

import copy
from pathlib import Path

import pytest
import torch
from PIL import Image

from src.dataset import make_loader
from src.presplit_dataset import (
    PresplitFaceDataset,
    dataset_fingerprint,
    resolve_data_root,
    split_counts,
)
from src.train_v2 import make_parser, smoke_test, validate_v2_resume


@pytest.fixture
def root(tmp_path):
    for split in ("train", "valid"):
        for label, folder in enumerate(("real", "fake")):
            directory = tmp_path / split / folder
            directory.mkdir(parents=True)
            for i in range(4):
                Image.new("RGB", (256, 256), (label * 100, i * 30, 80)).save(
                    directory / f"{i}.png"
                )
    # Test dizini training/validation kurulumu için gerekli değildir.
    return tmp_path


def test_label_mapping_samples_targets_and_getitem(root):
    for split in ("train", "valid"):
        dataset = PresplitFaceDataset(root, split)
        assert dataset.class_to_idx == {"real": 0, "fake": 1}
        assert dataset.classes == ["real", "fake"]
        for i, (path, label) in enumerate(dataset.samples):
            assert label == {"real": 0, "fake": 1}[Path(path).parent.name]
            image, target = dataset[i]
            assert target == label == dataset.targets[i]
            assert image.shape == (3, 224, 224) and torch.isfinite(image).all()


def test_loaders_preserve_split_membership_and_counts(root):
    datasets = {name: PresplitFaceDataset(root, name) for name in ("train", "valid")}
    assert split_counts(datasets) == {
        "train": {"REAL": 4, "FAKE": 4, "total": 8},
        "valid": {"REAL": 4, "FAKE": 4, "total": 8},
    }
    train_paths = {path for path, _ in datasets["train"].samples}
    val_paths = {path for path, _ in datasets["valid"].samples}
    assert not train_paths.intersection(val_paths)
    loader = make_loader(datasets["train"], batch_size=8, workers=0, shuffle=True)
    images, labels = next(iter(loader))
    assert images.shape == (8, 3, 224, 224)
    assert labels.tolist().count(0) == labels.tolist().count(1) == 4


def test_test_folder_never_traversed(root, monkeypatch):
    (root / "test").mkdir()
    original = Path.iterdir

    def guarded(path):
        if path == root / "test":
            raise AssertionError("Test klasörüne erişildi.")
        return original(path)

    monkeypatch.setattr(Path, "iterdir", guarded)
    datasets = {name: PresplitFaceDataset(root, name) for name in ("train", "valid")}
    fingerprint = dataset_fingerprint(datasets)
    (root / "test" / "ignored.txt").write_text(
        "Test içeriği fingerprint'i değiştirmemeli."
    )
    assert dataset_fingerprint(datasets) == fingerprint
    with pytest.raises(ValueError):
        PresplitFaceDataset(root, "test")


def test_fingerprint_detects_train_content_change(root):
    datasets = {name: PresplitFaceDataset(root, name) for name in ("train", "valid")}
    before = dataset_fingerprint(datasets)
    Image.new("RGB", (256, 256), "red").save(root / "train" / "real" / "0.png")
    assert dataset_fingerprint(datasets) != before


def test_alias_uses_existing_folder_without_renaming(tmp_path):
    alternative = tmp_path / "real-vs-fake"
    alternative.mkdir()
    assert resolve_data_root(tmp_path / "real_vs_fake") == alternative
    assert not (tmp_path / "real_vs_fake").exists()


def test_unexpected_classes_rejected(root):
    (root / "train" / "other").mkdir()
    with pytest.raises(ValueError):
        PresplitFaceDataset(root, "train")


def test_v2_cli_defaults_and_explicit_modes():
    args = make_parser().parse_args(["--run-training"])
    assert (args.epochs, args.batch_size, args.workers, args.seed, args.patience) == (
        10,
        8,
        0,
        42,
        3,
    )
    assert args.learning_rate == 3e-4 and args.device == "cuda"
    benchmark = make_parser().parse_args(
        ["--run-training", "--epochs", "1", "--resume", "auto"]
    )
    assert benchmark.epochs == 1 and benchmark.resume == "auto"
    with pytest.raises(SystemExit):
        make_parser().parse_args([])


def test_resume_rejects_old_model_changed_mapping_and_content(root):
    config = {"optimizer": "AdamW", "amp": True}
    state = {
        "pipeline": "presplit_image_v2",
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
    validate_v2_resume(state, "fp", config, root)
    for key, value in (
        ("pipeline", "original"),
        ("class_names", ["FAKE", "REAL"]),
        ("dataset_fingerprint", "changed"),
    ):
        invalid = copy.deepcopy(state)
        invalid[key] = value
        with pytest.raises(ValueError):
            validate_v2_resume(invalid, "fp", config, root)


def test_cuda_smoke_rejects_cpu_before_loading_model():
    with pytest.raises(ValueError, match="CUDA"):
        smoke_test(None, torch.device("cpu"))
