"""Optimizasyon/eğitim çalıştırmadan veri sızıntısı ve resume durumu testleri."""

import random
import shutil
from dataclasses import asdict

import numpy as np
import pytest
import torch
from PIL import Image

from src.dataset import (
    CLASS_NAMES,
    FaceDataset,
    audit_dataset,
    make_loader,
    seed_everything,
)
from src.train import (
    EarlyStopping,
    atomic_copy,
    atomic_save,
    capture_rng,
    calculate_timing,
    restore_rng,
    select_resume,
    validate_resume,
)


def test_epoch_timing_seconds_minutes_and_eta():
    timing = calculate_timing([], 120.0, 9)
    assert timing == {
        "epoch_duration": 120.0,
        "epoch_duration_minutes": 2.0,
        "total_training_duration": 120.0,
        "total_training_duration_minutes": 2.0,
        "eta": 1080.0,
        "eta_minutes": 18.0,
    }


def test_timing_accumulates_resume_history_and_uses_average_for_eta():
    timing = calculate_timing(
        [{"epoch_duration": 120.0}, {"epoch_duration": 180.0}], 60.0, 7
    )
    assert timing["total_training_duration"] == 360.0
    assert timing["total_training_duration_minutes"] == 6.0
    assert timing["eta"] == 840.0
    assert timing["eta_minutes"] == 14.0


def test_legacy_history_without_timing_remains_compatible():
    assert calculate_timing([{"epoch": 1, "train_loss": 0.5}], 90.0, 8)["eta"] == 720.0


@pytest.mark.parametrize("remaining", [0, -1])
def test_timing_eta_zero_at_completion_or_early_stopping(remaining):
    assert calculate_timing([{"epoch_duration": 120.0}], 180.0, remaining)["eta"] == 0.0


@pytest.fixture
def root(tmp_path):
    for label, folder in enumerate(("Real", "Fake")):
        (tmp_path / folder).mkdir()
        for index in range(20):
            Image.new("RGB", (12, 12), (index, label * 80, 150)).save(
                tmp_path / folder / f"{index}.png"
            )
    return tmp_path


def test_audit_preserves_duplicates_and_augmentation_families(root):
    shutil.copyfile(root / "Real/0.png", root / "Real/copy.png")
    for index in range(5):
        Image.new("RGB", (12, 12), (index + 40, 50, 150)).save(
            root / "Real" / f"real_99_aug_{index}.png"
        )
    report = audit_dataset(root)
    assert not report["errors"] and not report["path_errors"]
    assert not report["cross_split_groups"]
    assert report["augmentation_images"] == 5
    membership = {
        path: name for name, subset in report["splits"].items() for path, _ in subset
    }
    assert membership["Real/0.png"] == membership["Real/copy.png"]
    assert len({membership[f"Real/real_99_aug_{index}.png"] for index in range(5)}) == 1
    assert sum(part["total"] for part in report["split_counts"].values()) == 46
    assert report["splits"] == audit_dataset(root)["splits"]
    assert (root / "Real/copy.png").exists()
    assert report["source_identity_verified"] is False


def test_conflicting_duplicate_labels_block_split(root):
    shutil.copyfile(root / "Real/0.png", root / "Fake/conflict.png")
    report = audit_dataset(root)
    assert "farklı sınıf" in report["split_error"]
    assert "splits" not in report


def test_audit_reports_corruption_and_content_changes(root):
    first = audit_dataset(root)
    Image.new("RGB", (12, 12), (250, 250, 250)).save(root / "Real/0.png")
    assert first["fingerprint"] != audit_dataset(root)["fingerprint"]
    (root / "Real/broken.png").write_bytes(b"broken")
    assert audit_dataset(root)["errors"][0]["path"] == "Real/broken.png"


def test_early_stopping_and_state_roundtrip():
    stopper = EarlyStopping(patience=3)
    assert not stopper.update(0.5)
    assert not stopper.update(0.5)
    assert not stopper.update(0.4)
    restored = EarlyStopping(**asdict(stopper))
    assert restored.update(0.49)
    assert not restored.update(0.6)
    assert restored.bad_epochs == 0


def test_atomic_checkpoint_failure_preserves_existing(tmp_path):
    path = tmp_path / "checkpoint.pth"
    atomic_save({"epoch": 1}, path)
    with pytest.raises((AttributeError, TypeError)):
        atomic_save({"invalid": lambda: None}, path)
    assert torch.load(path, weights_only=True)["epoch"] == 1
    assert not list(tmp_path.glob("*.tmp"))
    atomic_copy(path, tmp_path / "best_model.pth")
    assert torch.load(tmp_path / "best_model.pth", weights_only=True)["epoch"] == 1


def test_auto_resume_recovers_epoch_before_last_update(tmp_path):
    assert select_resume(tmp_path, "auto") is None
    atomic_save({"epoch": 1}, tmp_path / "last.pth")
    atomic_save({"epoch": 1}, tmp_path / "epoch_001.pth")
    atomic_save({"epoch": 2}, tmp_path / "epoch_002.pth")
    assert select_resume(tmp_path, "auto") == tmp_path / "epoch_002.pth"
    assert select_resume(tmp_path, None) is None


def test_resume_dataset_and_config_guards(root):
    audit = audit_dataset(root)
    config = {"batch_size": 8, "learning_rate": 3e-4}
    checkpoint = {
        "training_state_version": 1,
        "dataset_fingerprint": audit["fingerprint"],
        "splits": audit["splits"],
        "config": config,
        "class_names": list(CLASS_NAMES),
        "image_size": 224,
    }
    validate_resume(checkpoint, audit, config)
    with pytest.raises(ValueError, match="ayarları"):
        validate_resume(checkpoint, audit, {**config, "batch_size": 4})
    with pytest.raises(ValueError, match="içeriği"):
        validate_resume(checkpoint, {**audit, "fingerprint": "changed"}, config)
    with pytest.raises(ValueError, match="ayrımı"):
        validate_resume(checkpoint, {**audit, "splits": {}}, config)


def test_resume_restores_rng_and_adamw_state_without_training(root, tmp_path):
    seed_everything()
    samples = audit_dataset(root)["splits"]["train"]
    train_loader = make_loader(FaceDataset(root, samples, training=True), shuffle=True)
    val_loader = make_loader(FaceDataset(root, samples))
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    # AdamW durumunun serileştirilmesini doğrular; backward/step çağrılmaz.
    for parameter in model.parameters():
        optimizer.state[parameter] = {
            "step": torch.tensor(2.0),
            "exp_avg": torch.full_like(parameter, 0.1),
            "exp_avg_sq": torch.full_like(parameter, 0.2),
        }
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "rng": capture_rng(train_loader, val_loader),
    }
    path = tmp_path / "resume.pth"
    atomic_save(checkpoint, path)
    expected = (
        random.random(),
        np.random.random(),
        torch.rand(3),
        torch.randperm(len(samples), generator=train_loader.generator),
    )
    restored = torch.load(path, weights_only=True, map_location="cpu")
    new_model = torch.nn.Linear(2, 2)
    new_model.load_state_dict(restored["model"])
    new_optimizer = torch.optim.AdamW(new_model.parameters(), lr=1.0)
    new_optimizer.load_state_dict(restored["optimizer"])
    scaler.load_state_dict(restored["scaler"])
    restore_rng(restored["rng"], train_loader, val_loader)
    assert random.random() == expected[0]
    assert np.random.random() == expected[1]
    assert torch.equal(torch.rand(3), expected[2])
    assert torch.equal(
        torch.randperm(len(samples), generator=train_loader.generator), expected[3]
    )
    assert new_optimizer.param_groups[0]["lr"] == 3e-4
    for parameter in new_model.parameters():
        assert new_optimizer.state[parameter]["step"].item() == 2
        assert torch.all(new_optimizer.state[parameter]["exp_avg"] == 0.1)
