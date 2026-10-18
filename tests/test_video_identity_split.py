"""Code-component isolation, preserved holdouts, reusable cache and clipped updates."""

import copy
import json

import pytest
import torch
from torch import nn

from src.video_cache import audit_feature_cache
from src.video_identity_split import (
    build_identity_manifest,
    identity_components,
    overlap_details,
    partition_components,
)
from src.video_manifest import digest_json, family_info, load_manifest
from src.video_pipeline import FeatureCache, video_collate
from src.video_train import (
    VideoNumericalError,
    checked_clip_gradients,
    train_video_batch,
    training_class_balance,
)


def record(path):
    family, identities = family_info(path)
    return dict(
        path=path,
        label=int(path.startswith("Celeb-synthesis/")),
        family=family,
        identities=identities,
        size=1,
        mtime_ns=1,
    )


@pytest.fixture
def parent(tmp_path):
    records = [
        record(f"Celeb-real/id{i}_{clip:04d}.mp4")
        for i in range(4)
        for clip in range(2)
    ]
    records += [
        record(f"Celeb-synthesis/id{i}_id{j}_0001.mp4")
        for i, j in [(0, 1), (1, 0), (2, 3), (3, 2)]
    ]
    records += [record(f"YouTube-real/{i:05d}.mp4") for i in range(10)]
    protected = [
        record("Celeb-real/id0_0002.mp4"),
        record("Celeb-synthesis/id0_id1_0002.mp4"),
    ]
    quarantine = [record("Celeb-synthesis/id1_id0_0002.mp4")]
    result = dict(
        schema="celeb_df_video_v1",
        dataset_root=str(tmp_path),
        seed=42,
        validation_fraction=0.2,
        splits=dict(
            train=records[::2], val=records[1::2], test=protected, quarantine=quarantine
        ),
        class_names=["REAL", "FAKE"],
        official_sha256="protected",
        source_identity_verified=False,
    )
    result["fingerprint"] = digest_json(result)
    return result


def test_both_filename_codes_and_transitive_protected_edges_are_grouped(parent):
    components = identity_components(
        [r for rows in parent["splits"].values() for r in rows]
    )
    assert components == [["id0", "id1"], ["id2", "id3"]]
    # Even a held-out manipulation must not be ignored as a known relationship.
    parent["splits"]["test"].append(record("Celeb-synthesis/id1_id2_0009.mp4"))
    with pytest.raises(ValueError, match="bileşen"):
        partition_components(parent)


def test_new_split_preserves_all_records_and_holdouts_without_code_overlap(
    parent, tmp_path
):
    before = copy.deepcopy(parent)
    derived = build_identity_manifest(parent, tmp_path / "parent.json")
    assert parent == before
    for name in ("test", "quarantine"):
        assert derived["splits"][name] == parent["splits"][name]
    assert sorted(
        parent["splits"]["train"] + parent["splits"]["val"], key=lambda r: r["path"]
    ) == sorted(
        derived["splits"]["train"] + derived["splits"]["val"], key=lambda r: r["path"]
    )
    assert derived["identity_overlap_counts"]["train_val"] == 0
    assert derived["source_identity_verified"] is False
    assert "garantisi yok" in derived["limitation"]
    assert derived == build_identity_manifest(parent, tmp_path / "parent.json")
    for name in ("train", "val"):
        assert {r["label"] for r in derived["splits"][name]} == {0, 1}


@pytest.mark.parametrize("fraction", [0, 1, -0.1, float("nan")])
def test_invalid_split_fraction_rejected(parent, fraction):
    with pytest.raises(ValueError):
        partition_components(parent, fraction)


def test_role_overlap_audit_does_not_conflate_first_and_second_codes(parent):
    rows = overlap_details(parent)
    assert len(rows) == 4
    id0 = next(r for r in rows if r["code"] == "id0")
    assert id0["train"]["fake_first_code"] == 1
    assert id0["validation"]["fake_second_code"] == 1
    assert id0["examples"]["train"]["fake_first_code"][0].startswith(
        "Celeb-synthesis/id0_id1_"
    )


@pytest.mark.parametrize("mutation", ["holdout", "membership", "parent", "codes"])
def test_rehashed_tampered_manifest_still_rejected(parent, tmp_path, mutation):
    original = tmp_path / "parent.json"
    original.write_text(json.dumps(parent), encoding="utf-8")
    derived = build_identity_manifest(parent, original)
    path = tmp_path / "new.json"
    if mutation == "holdout":
        derived["splits"]["test"].pop()
    elif mutation == "membership":
        derived["splits"]["val"].append(derived["splits"]["train"].pop())
    elif mutation == "parent":
        derived["parent_fingerprint"] = "wrong"
    else:
        derived["identity_overlap_counts"]["train_val"] = 9
    derived.pop("fingerprint")
    derived["fingerprint"] = digest_json(derived)
    path.write_text(json.dumps(derived), encoding="utf-8")
    with pytest.raises(ValueError, match="politikası"):
        load_manifest(path, verify_inventory=False)


def test_identity_manifest_loader_reconstructs_from_protected_parent(parent, tmp_path):
    original = tmp_path / "parent.json"
    original.write_text(json.dumps(parent), encoding="utf-8")
    derived = build_identity_manifest(parent, original)
    path = tmp_path / "new.json"
    path.write_text(json.dumps(derived), encoding="utf-8")
    assert load_manifest(path, verify_inventory=False) == derived


def populate_cache(parent, tmp_path):
    cache = FeatureCache(tmp_path / "cache", {}, "frozen-backbone")
    for rows in (parent["splits"]["train"], parent["splits"]["val"]):
        for r in rows:
            path = tmp_path / r["path"]
            path.parent.mkdir(exist_ok=True)
            path.write_bytes(r["path"].encode())
            cache.write(
                r,
                tmp_path,
                torch.ones(3, 1280),
                dict(
                    status="accepted",
                    valid_frames=3,
                    failed_frames=[],
                    relative_path=r["path"],
                ),
            )
    return cache


def test_record_cache_keys_reusable_after_reshuffle_and_weights_from_new_train(
    parent, tmp_path
):
    cache = populate_cache(parent, tmp_path)
    old_files = {p.name: p.read_bytes() for p in cache.directory.glob("*.pt")}
    derived = build_identity_manifest(parent, tmp_path / "parent.json")
    audit = audit_feature_cache(derived, cache)
    balance = training_class_balance(audit["splits"]["train"]["accepted_records"])
    real = audit["splits"]["train"]["accepted_counts"]["REAL"]
    fake = audit["splits"]["train"]["accepted_counts"]["FAKE"]
    assert balance["class_weights"]["REAL"] == (real + fake) / (2 * real)
    assert audit["cross_split_identical_mp4"] == [] and audit["cross_split_codes"] == []
    assert {p.name: p.read_bytes() for p in cache.directory.glob("*.pt")} == old_files


def test_duplicate_mp4_bytes_across_split_rejected(parent, tmp_path):
    derived = build_identity_manifest(parent, tmp_path / "parent.json")
    target = [
        next(
            r for r in derived["splits"][name] if r["path"].startswith("YouTube-real/")
        )
        for name in ("train", "val")
    ]
    cache = FeatureCache(tmp_path / "cache", {}, "frozen-backbone")
    for rows in (parent["splits"]["train"], parent["splits"]["val"]):
        for r in rows:
            path = tmp_path / r["path"]
            path.parent.mkdir(exist_ok=True)
            path.write_bytes(b"same-source" if r in target else r["path"].encode())
            cache.write(
                r,
                tmp_path,
                torch.ones(3, 1280),
                dict(status="accepted", valid_frames=3, failed_frames=[]),
            )
    with pytest.raises(ValueError, match="içerik"):
        audit_feature_cache(derived, cache)


class LargeGradientVideo(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(()))

    def forward(self, features, lengths):
        return self.weight * features[:, 0, 0]


@pytest.mark.parametrize("scale", [3779.6, 10000.0])
def test_optimizer_receives_clipped_gradient_not_preclip_norm(scale):
    model = LargeGradientVideo()
    item = dict(
        record=dict(path="synthetic", label=0),
        values=torch.full((3, 1280), scale),
        report={},
    )
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    seen = []
    optimizer.register_step_pre_hook(
        lambda *args: seen.append(model.weight.grad.abs().item())
    )
    result = train_video_batch(
        model,
        video_collate([item]),
        torch.device("cpu"),
        optimizer,
        torch.amp.GradScaler("cuda", enabled=False),
        True,
        False,
    )
    assert result["gradient_norm_pre_clip"] == pytest.approx(scale / 2)
    assert result["gradient_norm"] == result["gradient_norm_pre_clip"]
    assert result["gradient_norm_post_clip"] <= 1.0001 and seen[0] <= 1.0001
    assert model.weight.item() == pytest.approx(-0.01)


def test_broken_clipping_is_detected_before_optimizer(monkeypatch):
    parameter = nn.Parameter(torch.ones(1))
    parameter.grad = torch.tensor([1889.8])
    monkeypatch.setattr(
        nn.utils, "clip_grad_norm_", lambda *args, **kwargs: torch.tensor(1889.8)
    )
    with pytest.raises(VideoNumericalError, match="after_clip"):
        checked_clip_gradients([parameter], {})


def test_new_identity_finetune_requires_fresh_head(monkeypatch, tmp_path):
    from src import video_finetune as module
    from src.video_finetune_model import PartialVideoClassifier

    monkeypatch.setattr(
        module,
        "load_manifest",
        lambda path: dict(
            schema="celeb_df_identity_components_v1",
            fingerprint="new",
            dataset_root=str(tmp_path),
        ),
    )
    monkeypatch.setattr(
        module,
        "PartialVideoClassifier",
        lambda **kwargs: PartialVideoClassifier(pretrained=False),
    )
    args = module.make_parser().parse_args(
        ["--run-training", "--accept-filename-families"]
    )
    with pytest.raises(ValueError, match="fresh-head"):
        module.prepare_context(args)
    args.fresh_head = True
    records = [dict(path="a", label=0), dict(path="b", label=1)]
    monkeypatch.setattr(
        module,
        "audit_feature_cache",
        lambda *args: dict(
            cache_fingerprint="cache",
            splits={name: dict(accepted_records=records) for name in ("train", "val")},
        ),
    )
    monkeypatch.setattr(
        module, "VerifiedImageDataset", lambda root, records, *args: records
    )

    def no_learned_checkpoint(*args, **kwargs):
        raise AssertionError("Old learned checkpoint must not be loaded")

    monkeypatch.setattr(torch, "load", no_learned_checkpoint)
    _, _, _, _, _, config, _ = module.prepare_context(args)
    assert config["initialization"] == "imagenet_b0_v1_random_gru_head"
    assert (
        config["initial_checkpoint"] is None
        and config["initial_checkpoint_epoch"] is None
    )
