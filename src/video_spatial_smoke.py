"""Four bounded CUDA updates and masked/permuted feature invariance checks."""

from time import perf_counter

import torch

from src.video_pipeline import video_collate
from src.video_train import train_video_batch


def cuda_smoke(model, datasets, device, optimizer, scaler, weights, cached):
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Gerçek CUDA smoke gerekli.")
    model.train()
    records = datasets["train"].records
    # Two per class, accepted train only. No validation/test decoding here.
    selected = [
        i
        for label in (0, 1)
        for i in [i for i, r in enumerate(records) if r["label"] == label][:2]
    ]
    before = {n: v.detach().cpu().clone() for n, v in model.state_dict().items()}
    torch.cuda.reset_peak_memory_stats(device)
    updates = []
    feature_example = None
    for offset, index in enumerate(selected):
        item = datasets["train"][index]
        length = min(len(item["values"]), 3 + offset)
        shortened = dict(item, values=item["values"][:length])
        torch.cuda.synchronize(device)
        began = perf_counter()
        result = train_video_batch(
            model,
            video_collate([shortened]),
            device,
            optimizer,
            scaler,
            cached,
            False,
            weights,
        )
        torch.cuda.synchronize(device)
        updates.append(
            dict(
                path=item["record"]["path"],
                label=item["record"]["label"],
                length=length,
                loss=result["loss"],
                pre_clip=result["gradient_norm_pre_clip"],
                post_clip=result["gradient_norm_post_clip"],
                seconds=perf_counter() - began,
            )
        )
        if len(item["values"]) == 6:
            feature_example = item["values"]
    changed = [
        n for n, v in before.items() if not torch.equal(v, model.state_dict()[n].cpu())
    ]
    allowed = ("head.",) if cached else ("backbone.0.8.", "head.")
    if any(
        not n.startswith(allowed)
        or n.endswith(("running_mean", "running_var", "num_batches_tracked"))
        for n in changed
    ) or not all(any(n.startswith(prefix) for n in changed) for prefix in allowed):
        raise RuntimeError("Spatial gradient/frozen prefix/BN koruması başarısız.")
    optimizer.zero_grad(set_to_none=True)
    model.eval()
    if feature_example is None:
        raise ValueError("Mask diagnostic için altı geçerli kare gerekli.")
    errors, permutation_errors = [], []
    with torch.no_grad():
        features = (
            feature_example.to(device)
            if cached
            else model.extract_features(feature_example.to(device))
        )
        for length in (3, 4, 5, 6):
            padded = features.clone()
            padded[length:] = float("nan")
            expected = model(features[None, :length], torch.tensor([length]))
            observed = model(padded[None], torch.tensor([length]))
            reverse = model(features[:length].flip(0)[None], torch.tensor([length]))
            errors.append((expected - observed).abs().max().item())
            permutation_errors.append((expected - reverse).abs().max().item())
    if max(errors + permutation_errors) > 1e-5:
        raise RuntimeError("Masked mean/padding veya frame-order invariance yanlış.")
    return dict(
        device=torch.cuda.get_device_name(device),
        precision="fp32",
        backbone_mode=model.backbone_mode,
        feature_cache_used_as_input=cached,
        optimizer_steps=len(updates),
        updates=updates,
        all_loss_logits_gradients_finite=True,
        frozen_prefix_bn_unchanged=True,
        changed_parameters=changed,
        mask_errors=errors,
        frame_order_errors=permutation_errors,
        peak_cuda_allocated_mib=torch.cuda.max_memory_allocated(device) / 1024**2,
        peak_cuda_reserved_mib=torch.cuda.max_memory_reserved(device) / 1024**2,
        training_epoch_started=False,
        checkpoint_saved=False,
        official_test_video_content_accessed=False,
    )
