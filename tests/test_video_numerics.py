"""Finite loss/gradient guards and CUDA FP16 scale overflow regression."""

import pytest
import torch

from src.video_model import VideoClassifier
from src.video_pipeline import video_collate
from src.video_train import (
    VideoNumericalError,
    make_parser,
    train_video_batch,
    weighted_video_loss,
)


def setup_step(device="cpu"):
    torch.set_num_threads(2)
    model = VideoClassifier(pretrained=False, hidden_size=8).to(device).train()
    items = [
        dict(
            record=dict(path=f"synthetic_{length}", label=label),
            values=torch.randn(length, 1280),
            report={},
        )
        for length, label in [(3, 0), (6, 1)]
    ]
    batch = video_collate(items)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=3e-4
    )
    return model, batch, optimizer


def test_fp32_is_default_and_amp_is_explicit():
    args = make_parser().parse_args(["--run-training", "--accept-filename-families"])
    assert args.precision == "fp32"
    assert (
        make_parser()
        .parse_args(
            ["--run-training", "--accept-filename-families", "--precision", "amp"]
        )
        .precision
        == "amp"
    )


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_logits_and_labels_rejected(value):
    with pytest.raises(ValueError, match="NaN/Inf"):
        weighted_video_loss(torch.tensor([value]), torch.zeros(1))
    with pytest.raises(ValueError, match="NaN/Inf"):
        weighted_video_loss(torch.zeros(1), torch.tensor([value]))


def test_bce_half_inputs_use_float32_and_large_logits_stay_finite():
    logits = torch.tensor([-1000, 1000], dtype=torch.float16, requires_grad=True)
    loss = weighted_video_loss(
        logits, torch.tensor([1.0, 0.0]), torch.tensor([2.682143, 0.614566])
    )
    assert loss.dtype == torch.float32 and torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(logits.grad).all()


@pytest.mark.parametrize(
    "stage", ["features", "logits", "loss", "gradients", "gradient_norm"]
)
def test_invalid_step_never_reaches_optimizer(monkeypatch, stage):
    model, batch, optimizer = setup_step()
    before = model.head.weight.detach().clone()
    calls = []
    monkeypatch.setattr(optimizer, "step", lambda *args: calls.append("step"))
    if stage == "features":
        batch["items"][0]["values"][0, 0] = float("nan")
    elif stage == "logits":
        monkeypatch.setattr(
            model, "forward", lambda *args: torch.full((2,), float("inf"))
        )
    elif stage == "loss":
        monkeypatch.setattr(
            "src.video_train.weighted_video_loss",
            lambda *args: torch.tensor(float("inf")),
        )
    elif stage == "gradients":
        model.head.weight.register_hook(
            lambda grad: torch.full_like(grad, float("inf"))
        )
    else:

        def invalid_norm(*args, **kwargs):
            assert kwargs["error_if_nonfinite"] is True
            raise RuntimeError("non-finite gradient norm")

        monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", invalid_norm)
    with pytest.raises(VideoNumericalError) as caught:
        train_video_batch(
            model,
            batch,
            torch.device("cpu"),
            optimizer,
            torch.amp.GradScaler("cuda", enabled=False),
            True,
            False,
        )
    assert caught.value.details["stage"] == stage
    assert not calls and torch.equal(before, model.head.weight)
    assert all(p.grad is None for p in model.parameters())


def test_unscale_precedes_strict_clipping_and_step(monkeypatch):
    model, batch, optimizer = setup_step()
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    events = []
    unscale, clip, step = (
        scaler.unscale_,
        torch.nn.utils.clip_grad_norm_,
        optimizer.step,
    )

    def checked_unscale(optimizer):
        events.append("unscale")
        return unscale(optimizer)

    def checked_clip(*args, **kwargs):
        assert events == ["unscale"] and kwargs["error_if_nonfinite"] is True
        events.append("clip")
        return clip(*args, **kwargs)

    def checked_step(*args, **kwargs):
        assert events == ["unscale", "clip"]
        events.append("step")
        return step(*args, **kwargs)

    monkeypatch.setattr(scaler, "unscale_", checked_unscale)
    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", checked_clip)
    monkeypatch.setattr(optimizer, "step", checked_step)
    result = train_video_batch(
        model, batch, torch.device("cpu"), optimizer, scaler, True, False
    )
    assert events == ["unscale", "clip", "step"] and result["finite"]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_weighted_real_amp_overflow_is_detected_and_fp32_updates():
    model, batch, optimizer = setup_step("cuda")
    with torch.no_grad():
        model.head.weight.zero_()
        model.head.bias.zero_()
    batch = video_collate([batch["items"][0]])
    weights = torch.tensor([2253 / 840, 2253 / 3666], device="cuda")
    calls = []
    optimizer.register_step_post_hook(lambda *args: calls.append("step"))
    with pytest.raises(VideoNumericalError, match="gradients"):
        train_video_batch(
            model,
            batch,
            torch.device("cuda"),
            optimizer,
            torch.amp.GradScaler("cuda", init_scale=65536),
            True,
            True,
            weights,
        )
    assert not calls and not optimizer.state
    for amp, scale in [(False, 1), (True, 1024)]:
        result = train_video_batch(
            model,
            batch,
            torch.device("cuda"),
            optimizer,
            torch.amp.GradScaler("cuda", enabled=amp, init_scale=scale),
            True,
            amp,
            weights,
        )
        assert result["finite"] and result["gradient_norm"] > 0
    assert calls == ["step", "step"]
    assert all(torch.isfinite(p).all() for p in model.parameters())
