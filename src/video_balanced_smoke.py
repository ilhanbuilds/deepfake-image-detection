"""Eight real CUDA updates in memory; no epoch or checkpoint is created."""

from itertools import islice
from pathlib import Path
from time import perf_counter

import torch

from src.video_finetune import load_balanced_reference, prepare_context, write_json
from src.video_finetune_model import make_optimizer
from src.video_pipeline import make_video_loader, video_collate
from src.video_sampling import (
    gradient_summary,
    make_balanced_video_loader,
    sampling_summary,
)
from src.video_train import train_video_batch


class RecordDataset:
    """Preview actual sampling without reading any videos or feature tensors."""

    def __init__(self, records):
        self.records = records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        # video_collate only tests None here; this preview never goes to a model.
        return dict(record=self.records[index], values=True, report={})


def run_balanced_smoke(args):
    if args.sampling != "balanced" or args.resume or args.device != "cuda":
        raise ValueError("Balanced smoke: --sampling balanced, CUDA, resume yok.")
    output = args.run_dir.resolve()
    allowed = Path("outputs/video_finetune_v1").resolve()
    if not output.is_relative_to(allowed) or output == allowed or output.exists():
        raise ValueError("Smoke çıktısı yeni ayrı video_finetune_v1 dizini olmalı.")
    if not torch.cuda.is_available():
        raise RuntimeError("Gerçek CUDA smoke gerekli; CUDA bulunamadı.")
    manifest, model, _, audit, _, config, datasets = prepare_context(args)
    config["device_type"] = "cuda"
    reference = load_balanced_reference(args, manifest, config)
    records = audit["splits"]["train"]["accepted_records"]
    preview = make_balanced_video_loader(RecordDataset(records), records)
    draws = [batch["items"][0]["record"] for batch in preview]
    preview_stats = sampling_summary(records, draws)
    # Actual eight first draws from an identically seeded FULL accepted train loader.
    loader = make_balanced_video_loader(datasets["train"], records)
    device = torch.device("cuda")
    model.to(device).train()
    before = {n: v.detach().cpu().clone() for n, v in model.state_dict().items()}
    optimizer = make_optimizer(model, args.backbone_lr, args.head_lr)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    torch.cuda.reset_peak_memory_stats(device)
    updates, loaded = [], []
    for index, batch in enumerate(islice(loader, 8)):
        item = batch["items"][0]
        record = item["record"]
        if record != draws[index]:
            raise RuntimeError("Seed42 sampler preview ile gerçek CUDA sıra farklı.")
        loaded.append(item)
        length = min(len(item["values"]), 3 + index % 4)
        shortened = dict(item, values=item["values"][:length])
        torch.cuda.synchronize(device)
        began = perf_counter()
        result = train_video_batch(
            model,
            video_collate([shortened]),
            device,
            optimizer,
            scaler,
            False,
            False,
            class_weights=None,
        )
        torch.cuda.synchronize(device)
        if any(
            p.grad is None or not torch.isfinite(p.grad).all()
            for p in model.backbone[0][-1].parameters()
        ):
            raise RuntimeError("features[8] sonlu gerçek görüntü gradient almadı.")
        updates.append(
            dict(
                path=record["path"],
                label=record["label"],
                length=length,
                loss=result["loss"],
                logit=result["logits"].item(),
                pre_clip=result["gradient_norm_pre_clip"],
                post_clip=result["gradient_norm_post_clip"],
                seconds=perf_counter() - began,
            )
        )
    after = model.state_dict()
    changed = [n for n, v in before.items() if not torch.equal(v, after[n].cpu())]
    if (
        any(not n.startswith(("backbone.0.8.", "gru.", "head.")) for n in changed)
        or any(
            n.endswith(("running_mean", "running_var", "num_batches_tracked"))
            for n in changed
        )
        or not all(
            any(n.startswith(prefix) for n in changed)
            for prefix in ("backbone.0.8.", "gru.", "head.")
        )
    ):
        raise RuntimeError(
            "Frozen prefix/BN veya son blok/GRU/head güncellemesi yanlış."
        )
    optimizer.zero_grad(set_to_none=True)
    model.eval()
    errors = []
    with torch.no_grad():
        item = next((row for row in loaded if len(row["values"]) == 6), None)
        if item is None:
            raise RuntimeError("Smoke masking kontrolü için altı geçerli kare gerekli.")
        features = model.extract_features(item["values"].to(device))
        for length in (3, 4, 5, 6):
            padded = features.clone()
            padded[length:] = float("nan")
            a = model(features[None, :length], torch.tensor([length]))
            b = model(padded[None], torch.tensor([length]))
            errors.append((a - b).abs().max().item())
    if max(errors) > 1e-5:
        raise RuntimeError("3–6 kare NaN padding mask testi başarısız.")
    val_loader = make_video_loader(datasets["val"], 1)
    if len(val_loader) != 602 or not torch.equal(
        val_loader.generator.get_state(), torch.Generator().manual_seed(42).get_state()
    ):
        raise RuntimeError("Validation roster/seed değişti.")
    output.mkdir(parents=True, exist_ok=False)
    report = dict(
        device=torch.cuda.get_device_name(device),
        precision="fp32",
        loss="unweighted_video_bce",
        loss_class_weights=None,
        manifest_fingerprint=manifest["fingerprint"],
        reference=reference,
        accepted_counts={
            name: audit["splits"][name]["accepted_counts"] for name in ("train", "val")
        },
        expected_epoch_optimizer_steps=len(loader),
        maximum_training_epochs=3,
        epoch1_sampling_preview=preview_stats,
        preview_only_no_full_epoch_decode=True,
        cuda_optimizer_steps=len(updates),
        updates=updates,
        clipping_by_class=gradient_summary(updates),
        padding_max_logit_errors=errors,
        all_loss_logits_gradients_finite=True,
        frozen_prefix_and_bn_unchanged=True,
        changed_parameters=changed,
        feature_cache_used_as_input=False,
        peak_cuda_allocated_mib=torch.cuda.max_memory_allocated(device) / 1024**2,
        peak_cuda_reserved_mib=torch.cuda.max_memory_reserved(device) / 1024**2,
        checkpoint_saved=False,
        training_epoch_started=False,
        official_test_video_content_accessed=False,
    )
    write_json(output / "smoke.json", report)
    write_json(output / "reference_config.json", config)
    print(
        f"CUDA smoke: {len(updates)} finite unweighted updates; frozen prefix/BN unchanged; {output / 'smoke.json'}"
    )
