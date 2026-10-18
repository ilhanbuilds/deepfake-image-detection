"""Video safety, temporal training and synthetic decoding; no real dataset training."""

import copy
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from src.video_evaluate import evaluate_video, load_video_checkpoint
from src.video_manifest import FOLDERS, audit, load_manifest, parse_official
from src.video_model import VideoClassifier
from src.video_pipeline import (
    FacePipeline,
    FeatureCache,
    make_video_loader,
    sample_indices,
)
from src.video_predict import predict_video
from src.video_preprocess import select_smoke
from src.video_train import make_parser, run_training, train_epoch, validate_resume


@pytest.fixture(scope="module")
def video_model():
    torch.set_num_threads(2)
    torch.manual_seed(42)
    return VideoClassifier(pretrained=False)


@pytest.fixture
def mini_root(tmp_path):
    for folder in FOLDERS:
        (tmp_path / folder).mkdir()
    for identity in range(4):
        for clip in range(4):
            (tmp_path / "Celeb-real" / f"id{identity}_{clip:04d}.mp4").write_bytes(
                b"synthetic real inventory"
            )
            for donor in range(2):
                (
                    tmp_path
                    / "Celeb-synthesis"
                    / f"id{identity}_id{donor + 5}_{clip:04d}.mp4"
                ).write_bytes(b"synthetic fake inventory")
    for clip in range(4):
        (tmp_path / "YouTube-real" / f"{clip:05d}.mp4").write_bytes(
            b"synthetic youtube inventory"
        )
    (tmp_path / "List_of_testing_videos.txt").write_text(
        "1 Celeb-real/id0_0000.mp4\n0 Celeb-synthesis/id1_id5_0000.mp4\n1 YouTube-real/00000.mp4\n",
        encoding="utf-8",
    )
    return tmp_path


def test_official_mapping_families_and_quarantine(mini_root):
    result = audit(mini_root)
    assert result == audit(mini_root)
    splits = result["splits"]
    assert result["split_counts"]["test"] == dict(total=3, REAL=2, FAKE=1)
    assert result["cross_split_families"] == []
    assert result["source_identity_verified"] is False
    assert len(splits["quarantine"]) == 4
    all_paths = [r["path"] for subset in splits.values() for r in subset]
    assert len(all_paths) == len(set(all_paths)) == 52
    family_sets = [
        {r["family"] for r in splits[name]} for name in ("train", "val", "test")
    ]
    assert not family_sets[0] & family_sets[1]
    assert not (family_sets[0] | family_sets[1]) & family_sets[2]
    assert {r["label"] for r in splits["val"]} == {0, 1}


@pytest.mark.parametrize(
    "line",
    [
        "0 Celeb-real/id0_0000.mp4",
        "1 Celeb-real/missing.mp4",
        "1 ../outside.mp4",
        "1  Celeb-real/id0_0000.mp4",
        "REAL Celeb-real/id0_0000.mp4",
        "1 Celeb-real/id0_0000.mp4\n1 Celeb-real/id0_0000.mp4",
    ],
)
def test_bad_official_entries_rejected(mini_root, line):
    records = {
        r["path"]: r for subset in audit(mini_root)["splits"].values() for r in subset
    }
    (mini_root / "List_of_testing_videos.txt").write_text(line, encoding="utf-8")
    with pytest.raises(ValueError):
        parse_official(mini_root, records)


def test_inventory_changes_block_saved_manifest(mini_root, tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(audit(mini_root)), encoding="utf-8")
    assert load_manifest(manifest)["class_names"] == ["REAL", "FAKE"]
    (mini_root / "Celeb-real/id2_0000.mp4").write_bytes(b"changed content")
    with pytest.raises(ValueError, match="değişti"):
        load_manifest(manifest)


def test_missing_source_candidate_blocks_audit(mini_root):
    (mini_root / "Celeb-synthesis/id99_id0_0000.mp4").write_bytes(b"invalid inventory")
    with pytest.raises(ValueError, match="kaynak adayı"):
        audit(mini_root)


def test_equal_spacing_unique_and_short_rejection():
    assert sample_indices(31) == [0, 6, 12, 18, 24, 30]
    assert sample_indices(6) == list(range(6))
    with pytest.raises(ValueError):
        sample_indices(5)


def test_actual_decoder_reports_no_face_without_fallback(tmp_path, video_model):
    cv = pytest.importorskip("cv2")
    path = tmp_path / "blank.avi"
    writer = cv.VideoWriter(str(path), cv.VideoWriter_fourcc(*"MJPG"), 10, (80, 80))
    assert writer.isOpened()
    for _ in range(11):
        writer.write(np.zeros((80, 80, 3), np.uint8))
    writer.release()
    pipeline = FacePipeline()
    frames, report, _ = pipeline.read(path)
    assert frames is None and report["valid_frames"] == 0
    assert report["selected_indices"] == [0, 2, 4, 6, 8, 10]
    assert len(report["failed_frames"]) == 6
    assert {r["reason"] for r in report["failed_frames"]} == {"face_not_detected"}
    result = predict_video(video_model, path, torch.device("cpu"), pipeline)
    assert result["prediction"] is result["fake_probability"] is None


def test_crop_rgb_geometry_and_consistent_track(monkeypatch):
    pytest.importorskip("cv2")
    pipeline = FacePipeline()
    monkeypatch.setattr(
        pipeline,
        "detector",
        SimpleNamespace(
            detectMultiScale=lambda *a, **k: np.array(
                [[30, 30, 30, 30], [1, 1, 10, 10]]
            )
        ),
    )
    bgr = np.zeros((100, 100, 3), np.uint8)
    bgr[:, :, 2] = 255
    crop, box, reason = pipeline.crop(bgr, None)
    assert box == (30, 30, 30, 30) and reason is None
    assert crop.getpixel((0, 0)) == (255, 0, 0)
    assert pipeline.transform(crop).shape == (3, 160, 160)
    crop, previous, reason = pipeline.crop(bgr, (70, 70, 20, 20))
    assert crop is None and previous == (70, 70, 20, 20) and reason == "face_track_lost"


def test_missing_video_rejected(tmp_path):
    pytest.importorskip("cv2")
    frames, report, _ = FacePipeline().read(tmp_path / "missing.mp4")
    assert frames is None and report["error"] == "video_open_failed"


def test_successful_decoder_keeps_six_chronological_crops(tmp_path, monkeypatch):
    cv = pytest.importorskip("cv2")
    path = tmp_path / "frames.avi"
    writer = cv.VideoWriter(str(path), cv.VideoWriter_fourcc(*"MJPG"), 10, (80, 80))
    assert writer.isOpened()
    for index in range(11):
        writer.write(np.full((80, 80, 3), index * 20, np.uint8))
    writer.release()
    pipeline = FacePipeline()
    monkeypatch.setattr(
        pipeline,
        "detector",
        SimpleNamespace(detectMultiScale=lambda *a, **k: np.array([[20, 20, 40, 40]])),
    )
    frames, report, crops = pipeline.read(path)
    assert frames.shape == (6, 3, 160, 160)
    assert report["status"] == "accepted" and not report["failed_frames"]
    assert report["valid_indices"] == [0, 2, 4, 6, 8, 10]
    brightness = [np.asarray(crop).mean() for crop in crops]
    assert brightness == sorted(brightness) and len(set(brightness)) == 6


def test_nonfinite_inference_fails_closed(video_model, monkeypatch):
    model = copy.deepcopy(video_model)
    monkeypatch.setattr(model, "forward", lambda *a: torch.tensor([float("nan")]))
    pipeline = SimpleNamespace(
        read=lambda path: (torch.zeros(3, 3, 160, 160), dict(status="accepted"), [])
    )
    with pytest.raises(RuntimeError, match="tahmin üretilmedi"):
        predict_video(model, None, torch.device("cpu"), pipeline)


def test_backbone_freeze_bn_and_video_gradient(video_model):
    model = copy.deepcopy(video_model).train()
    assert not model.backbone.training
    assert all(not p.requires_grad for p in model.backbone.parameters())
    before = {
        name: tensor.clone() for name, tensor in model.backbone.state_dict().items()
    }
    features = model.extract_features(torch.randn(6, 3, 160, 160))
    assert features.shape == (6, 1280) and not features.requires_grad
    logits = model(features.unsqueeze(0), torch.tensor([6]))
    assert logits.shape == (1,)
    nn.functional.binary_cross_entropy_with_logits(logits, torch.ones(1)).backward()
    assert (
        model.gru.weight_ih_l0.grad is not None and model.head.weight.grad is not None
    )
    assert all(p.grad is None for p in model.backbone.parameters())
    assert all(
        torch.equal(tensor, before[name])
        for name, tensor in model.backbone.state_dict().items()
    )


def test_temporal_order_and_padding_mask(video_model):
    video_model.eval()
    features = torch.randn(1, 6, 1280)
    with torch.no_grad():
        original = video_model(features, torch.tensor([6]))
        reversed_order = video_model(features.flip(1), torch.tensor([6]))
        padded = torch.cat([features, torch.ones(1, 2, 1280) * 1000], dim=1)
        padded_logits = video_model(padded, torch.tensor([6]))
    assert not torch.allclose(original, reversed_order)
    assert torch.allclose(original, padded_logits)


def test_all_time_steps_receive_gradient(video_model):
    features = torch.randn(1, 6, 1280, requires_grad=True)
    video_model(features, torch.tensor([6])).sum().backward()
    assert torch.all(features.grad.abs().sum(dim=2) > 0)


@pytest.mark.parametrize(
    "lengths", [torch.tensor([0]), torch.tensor([7]), torch.tensor([6, 6])]
)
def test_invalid_lengths_rejected(video_model, lengths):
    with pytest.raises(ValueError):
        video_model(torch.zeros(1, 6, 1280), lengths)


def synthetic_items():
    items = []
    for index, label in enumerate((0, 1, 0, 1)):
        report = dict(
            status="accepted",
            path=f"synthetic{index}",
            valid_frames=6,
            failed_frames=[],
        )
        items.append(
            dict(
                record=dict(path=f"synthetic{index}", label=label),
                values=torch.full((6, 1280), float(label)),
                report=report,
            )
        )
    items.append(
        dict(
            record=dict(path="failed", label=1),
            values=None,
            report=dict(
                status="rejected",
                failed_frames=[dict(index=0, reason="face_not_detected")],
                valid_frames=0,
            ),
        )
    )
    return items


def synthetic_cache_audit(manifest, cache):
    """Training integration uses synthetic features; real cache audit tested separately."""
    return dict(
        cache_fingerprint="synthetic_cache_fingerprint",
        splits={
            name: dict(
                accepted_records=records,
                rejected_reports=[],
                accepted=len(records),
                rejected=0,
            )
            for name, records in manifest["splits"].items()
        },
    )


def test_video_metrics_and_failure_coverage(video_model):
    loader = make_video_loader(synthetic_items())
    result = evaluate_video(video_model, loader, torch.device("cpu"), cached=True)
    assert result["coverage"]["attempted_videos"] == 5
    assert result["coverage"]["evaluated_videos"] == 4
    assert len(result["predictions"]) == 4
    assert result["coverage"]["failed_frame_count"] == 1
    assert sum(map(sum, result["metrics"]["confusion_matrix"])) == 4
    assert result["metrics"]["positive_class"] == "FAKE"
    assert all(
        key in result["metrics"]
        for key in ("accuracy", "precision", "recall", "f1", "roc_auc")
    )


def test_training_updates_head_only_and_reports_failures(video_model):
    model = copy.deepcopy(video_model)
    before = model.head.weight.detach().clone()
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=1e-3
    )
    result = train_epoch(
        model,
        make_video_loader(synthetic_items()),
        torch.device("cpu"),
        optimizer,
        torch.amp.GradScaler("cuda", enabled=False),
        True,
        False,
        1,
    )
    assert result["coverage"]["evaluated_videos"] == 4
    assert not torch.equal(before, model.head.weight)
    assert all(p.grad is None for p in model.backbone.parameters())


def test_cache_content_and_signature_safety(tmp_path):
    record = dict(path="source.mp4", label=1, size=3)
    (tmp_path / "source.mp4").write_bytes(b"abc")
    cache = FeatureCache(tmp_path / "cache", dict(image_size=160), "backbone1")
    features = torch.ones(6, 1280)
    report = dict(valid_frames=6, status="accepted", failed_frames=[])
    cache.write(record, tmp_path, features, report)
    assert torch.equal(cache.read(record, tmp_path)["values"], features)
    with pytest.raises(FileExistsError):
        cache.write(record, tmp_path, features, report)
    other = FeatureCache(tmp_path / "cache", dict(image_size=192), "backbone1")
    with pytest.raises(FileNotFoundError):
        other.read(record, tmp_path)
    other = FeatureCache(tmp_path / "cache", dict(image_size=160), "backbone2")
    assert other.path(record) != cache.path(record)
    (tmp_path / "source.mp4").write_bytes(b"changed source")
    with pytest.raises(ValueError, match="uyumsuz"):
        cache.read(record, tmp_path)


def test_cache_rejection_retained(tmp_path):
    record = dict(path="source.mp4", label=0)
    (tmp_path / "source.mp4").write_bytes(b"a")
    cache = FeatureCache(tmp_path / "cache", {}, "backbone")
    report = dict(status="rejected", valid_frames=0, failed_frames=[])
    cache.write(record, tmp_path, None, report)
    assert cache.read(record, tmp_path)["values"] is None


def test_smoke_selection_never_selects_test(mini_root):
    manifest = audit(mini_root)
    selected = select_smoke(manifest["splits"]["train"])
    assert len(selected) == 4
    assert not {r["path"] for r in selected} & {
        r["path"] for r in manifest["splits"]["test"]
    }


def test_training_requires_explicit_flags():
    with pytest.raises(SystemExit):
        make_parser().parse_args([])
    with pytest.raises(SystemExit):
        make_parser().parse_args(["--run-training"])


def test_image_checkpoint_rejected(tmp_path):
    path = tmp_path / "image.pth"
    torch.save(dict(class_names=["REAL", "FAKE"], image_size=224), path)
    with pytest.raises(ValueError, match="fotoğraf checkpoint"):
        load_video_checkpoint(path, torch.device("cpu"))


def test_resume_config_and_manifest_guards():
    with pytest.raises(ValueError):
        validate_resume(dict(pipeline="image_v3"), dict(fingerprint="x"), {})
    with pytest.raises(ValueError, match="eksik"):
        validate_resume(
            dict(
                pipeline="video_v1_frozen_gru",
                class_names=["REAL", "FAKE"],
                manifest_fingerprint="x",
                config={},
            ),
            dict(fingerprint="x"),
            {},
        )


def test_epoch_checkpoint_resume_matches_uninterrupted(
    monkeypatch, tmp_path, video_model
):
    import src.video_train as training

    pytest.importorskip("cv2")
    records = [item["record"] for item in synthetic_items()[:4]]
    manifest = dict(
        fingerprint="synthetic_manifest",
        dataset_root=str(tmp_path),
        splits=dict(train=records, val=records),
    )
    monkeypatch.setattr(training, "load_manifest", lambda path: manifest)
    monkeypatch.setattr(training, "audit_feature_cache", synthetic_cache_audit)
    monkeypatch.setattr(
        training,
        "VideoClassifier",
        lambda **kwargs: VideoClassifier(
            pretrained=False, hidden_size=kwargs["hidden_size"]
        ),
    )
    # Synthetic frozen features exercise real GRU optimizer/checkpoint/RNG code only.
    monkeypatch.setattr(training, "FeatureDataset", lambda *args: synthetic_items())

    def args(directory, epochs, resume=None):
        return SimpleNamespace(
            epochs=epochs,
            hidden_size=8,
            patience=3,
            learning_rate=3e-4,
            manifest=tmp_path / "unused.json",
            image_size=160,
            device="cpu",
            run_dir=directory,
            resume=resume,
            batch_size=2,
            cache_dir=tmp_path / "synthetic_cache",
        )

    uninterrupted = tmp_path / "uninterrupted"
    resumed = tmp_path / "resumed"
    run_training(args(uninterrupted, 2))
    run_training(args(resumed, 1))
    original_epoch = (resumed / "checkpoints/epoch_001.pth").read_bytes()
    run_training(args(resumed, 2, "auto"))
    assert (resumed / "checkpoints/epoch_001.pth").read_bytes() == original_epoch
    first = torch.load(uninterrupted / "checkpoints/epoch_002.pth", weights_only=True)
    second = torch.load(resumed / "checkpoints/epoch_002.pth", weights_only=True)
    assert first["best_epoch"] == second["best_epoch"]
    assert "epochs" not in first["config"]  # Total budget can grow on resume.

    def assert_same_state(expected, actual):
        if isinstance(expected, torch.Tensor):
            assert torch.equal(expected, actual)
        elif isinstance(expected, dict):
            assert expected.keys() == actual.keys()
            for key in expected:
                assert_same_state(expected[key], actual[key])
        elif isinstance(expected, (list, tuple)):
            assert len(expected) == len(actual)
            for left, right in zip(expected, actual, strict=True):
                assert_same_state(left, right)
        else:
            assert expected == actual

    for name in (
        "optimizer_state_dict",
        "scaler_state_dict",
        "rng_state",
        "early_stopping",
    ):
        assert_same_state(first[name], second[name])
    assert first["config"]["selection"] == "validation_macro_F1"
    assert first["best_macro_f1"] == second["best_macro_f1"]
    record = json.loads(
        (resumed / "experiment_record.json").read_text(encoding="utf-8")
    )
    assert record["results"]["best_epoch"] == second["best_epoch"]
    assert all(
        torch.equal(tensor, second["model_state_dict"][name])
        for name, tensor in first["model_state_dict"].items()
    )
    with pytest.raises(FileExistsError):
        run_training(args(resumed, 2))


def test_oom_does_not_save_partial_epoch(monkeypatch, tmp_path):
    import src.video_train as training

    pytest.importorskip("cv2")
    monkeypatch.setattr(training, "audit_feature_cache", synthetic_cache_audit)
    monkeypatch.setattr(
        training,
        "load_manifest",
        lambda path: dict(
            fingerprint="synthetic",
            dataset_root=str(tmp_path),
            splits=dict(train=[dict(label=0), dict(label=1)], val=[]),
        ),
    )
    monkeypatch.setattr(
        training,
        "VideoClassifier",
        lambda **kwargs: VideoClassifier(pretrained=False, hidden_size=8),
    )
    monkeypatch.setattr(
        training,
        "make_video_loader",
        lambda *a, **k: make_video_loader(synthetic_items()),
    )

    def oom(*args):
        raise torch.cuda.OutOfMemoryError("synthetic CUDA OOM")

    monkeypatch.setattr(training, "train_epoch", oom)
    args = SimpleNamespace(
        epochs=1,
        hidden_size=8,
        patience=3,
        learning_rate=3e-4,
        manifest=tmp_path / "unused",
        image_size=160,
        device="cpu",
        run_dir=tmp_path / "run",
        resume=None,
        batch_size=2,
        cache_dir=tmp_path / "synthetic_cache",
    )
    with pytest.raises(RuntimeError, match="kesildi"):
        run_training(args)
    assert not list((args.run_dir / "checkpoints").glob("*.pth"))
    assert (
        json.loads((args.run_dir / "interrupted.json").read_text(encoding="utf-8"))[
            "reason"
        ]
        == "OutOfMemoryError"
    )
