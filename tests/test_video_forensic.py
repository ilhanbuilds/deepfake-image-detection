"""Safety and mathematical contracts; no real dataset/test-set access."""

import copy
import sys

import numpy as np
import pytest
import torch
from torch import nn

from src.video_forensic import accepted_roster, frame_scores, score_faces, summarize
from src.video_forensic_train import PIPELINE, train_group, validate_state
from src.video_forensic_train import main as training_main
from src.video_forensic_robustness import CONDITIONS, corrupt
from src.video_quality_fusion import fuse, fusion_weights, relative_quality


def report(sharpness=(1.0, 2.0, 3.0)):
    return dict(
        status="accepted",
        frames=[
            dict(
                index=i,
                faces=[
                    dict(
                        crop_index=i,
                        bbox=[0, 0, 80, 80],
                        crop_size=[101, 101],
                        sharpness=s,
                    )
                ],
            )
            for i, s in enumerate(sharpness)
        ],
    )


@pytest.mark.parametrize("split", ["test", "quarantine", "official_test", "all"])
def test_test_roster_locked_without_io(split):
    with pytest.raises(ValueError, match="locked"):
        accepted_roster(split)


def test_score_is_fake_softmax_column_and_uint8_contract():
    class Probe(nn.Module):
        def forward(self, x):
            assert x.dtype == torch.float32
            assert x.min() == 0 and x.max() == 1
            return torch.tensor([[0.0, np.log(3.0)]], dtype=torch.float32)

    pixels = torch.zeros((1, 3, 380, 380), dtype=torch.uint8)
    pixels[0, 0, 0, 0] = 255
    assert score_faces(Probe(), pixels, "cpu") == pytest.approx([0.75])


def test_multiface_max_then_video_mean():
    r = report()
    r["frames"][0]["faces"].append(dict(r["frames"][0]["faces"][0], crop_index=3))
    assert frame_scores([0.1, 0.2, 0.3, 0.9], r) == [0.9, 0.2, 0.3]


def test_nonfinite_model_output_fails():
    class Bad(nn.Module):
        def forward(self, x):
            return torch.full((1, 2), float("nan"))

    with pytest.raises(ValueError, match="logits"):
        score_faces(Bad(), torch.zeros((1, 3, 380, 380), dtype=torch.uint8), "cpu")


def test_quality_equal_values_return_mean():
    r = report((2, 2, 2))
    assert fusion_weights(r) == pytest.approx([1 / 3] * 3)
    assert fuse([0.1, 0.5, 0.9], r)[0] == pytest.approx(0.5)


@pytest.mark.parametrize("n", [1, 3, 4, 6, 32])
def test_quality_weights_are_bounded_and_never_drop_frames(n):
    weights = fusion_weights(report(tuple(range(n))))
    assert weights.sum() == pytest.approx(1)
    assert np.all(weights >= 2 / (3 * n) - 1e-12)
    assert np.all(weights <= 2 / n + 1e-12)


def test_quality_invariant_to_common_sharpness_shift_and_scale():
    assert relative_quality([1, 2, 8]) == pytest.approx(
        relative_quality([102, 104, 116])
    )


def test_fake_score_does_not_control_quality_weights():
    r = report()
    assert fuse([0.01, 0.1, 0.9], r)[1] == fuse([0.99, 0.5, 0.01], r)[1]


@pytest.mark.parametrize("scores", [[1, 2, 3], [float("nan"), 0.1, 0.2], [0.2]])
def test_invalid_probabilities_rejected(scores):
    with pytest.raises(ValueError):
        fuse(scores, report())


def test_coverage_keeps_failed_video_visible():
    rows = [
        dict(
            path="Celeb-real/a",
            target=0,
            status="accepted",
            prediction=0,
            fake_probability=0.1,
        ),
        dict(
            path="Celeb-synthesis/b",
            target=1,
            status="accepted",
            prediction=1,
            fake_probability=0.9,
        ),
        dict(path="Celeb-real/c", target=0, status="rejected"),
    ]
    result = summarize(rows, 3)
    assert result["coverage"] == 2 / 3 and len(result["rejected"]) == 1


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(3, 2)

    def forward(self, x):
        return self.fc(x.mean((2, 3)))


def test_streamed_gradient_matches_exact_video_balanced_objective():
    torch.manual_seed(42)
    model = Tiny()
    reference = copy.deepcopy(model)
    examples = [
        (torch.randint(0, 256, (3, 3, 4, 4), dtype=torch.uint8), report(), label)
        for label in (0, 1)
    ]
    weights = torch.tensor([2.0, 0.6])
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    expected_optimizer = torch.optim.SGD(reference.parameters(), lr=0.01)
    loss = sum(
        nn.functional.cross_entropy(
            reference(p.float() / 255), torch.full((3,), label, dtype=torch.long)
        )
        * weights[label]
        / 2
        for p, _, label in examples
    )
    loss.backward()
    nn.utils.clip_grad_norm_(reference.parameters(), 1, error_if_nonfinite=True)
    expected_optimizer.step()
    result = train_group(model, optimizer, examples, weights, "cpu")
    assert result["loss"] == pytest.approx(loss.item())
    for actual, expected in zip(
        model.parameters(), reference.parameters(), strict=True
    ):
        torch.testing.assert_close(actual, expected)


def test_rejected_video_cannot_be_silently_trained():
    model = Tiny()
    with pytest.raises(ValueError, match="Rejected"):
        train_group(
            model,
            torch.optim.SGD(model.parameters(), lr=0.1),
            [(torch.zeros(1), dict(status="rejected"), 0)],
            torch.ones(2),
            "cpu",
        )


def test_resume_rejects_config_or_missing_state():
    config = dict(seed=42)
    state = dict(
        pipeline=PIPELINE,
        config=config,
        model_state_dict={},
        optimizer_state_dict={},
        rng_state={},
        history=[],
        epoch=1,
        early_stopping={},
        best_epoch=1,
        best_score=0.5,
        validation={},
    )
    validate_state(state, config)
    with pytest.raises(ValueError):
        validate_state(state, dict(seed=43))
    del state["rng_state"]
    with pytest.raises(ValueError):
        validate_state(state, config)


@pytest.mark.parametrize("condition", CONDITIONS)
def test_corruption_reproducible_label_free_and_preserves_input(condition):
    rng = torch.Generator().manual_seed(42)
    pixels = torch.randint(0, 256, (4, 3, 380, 380), dtype=torch.uint8, generator=rng)
    original = pixels.clone()
    metadata = report((1, 2, 3, 4))
    original_report = copy.deepcopy(metadata)
    a, ar = corrupt(pixels, metadata, condition, 42, "video.mp4")
    b, br = corrupt(pixels, metadata, condition, 42, "video.mp4")
    assert torch.equal(a, b) and ar == br
    assert torch.equal(pixels, original) and metadata == original_report
    assert len(ar["frames"]) == (3 if condition == "drop_frame" else 4)
    assert a.dtype == torch.uint8 and a.shape == pixels.shape


def test_training_requires_explicit_action_before_any_data_access(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        ["video_forensic_train", "--run-dir", "outputs/video_research_v1/finetune/new"],
    )
    with pytest.raises(SystemExit) as error:
        training_main()
    assert error.value.code == 2


def test_training_cannot_overwrite_existing_run(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    existing = tmp_path / "outputs/video_research_v1/finetune/keep"
    existing.mkdir(parents=True)
    marker = existing / "checkpoint.txt"
    marker.write_text("protected")
    monkeypatch.setattr(
        sys,
        "argv",
        ["video_forensic_train", "--run-training", "--run-dir", str(existing)],
    )
    with pytest.raises(FileExistsError, match="protected"):
        training_main()
    assert marker.read_text() == "protected"
