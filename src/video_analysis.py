"""Read-only V1 checkpoint/cache analysis and a single short train/val baseline fit."""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from time import perf_counter
from uuid import uuid4

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.dataset import seed_everything
from src.model import resolve_device
from src.video_baseline import (
    baseline_predictions,
    export_baseline,
    fit_baseline,
    pooled_matrix,
    temporal_mean,
)
from src.video_cache import audit_feature_cache
from src.video_evaluate import (
    calculate_video_metrics,
    evaluate_video,
    load_video_checkpoint,
)
from src.video_manifest import DEFAULT_MANIFEST, load_manifest
from src.video_pipeline import (
    FacePipeline,
    FeatureCache,
    VideoSettings,
    file_sha256,
    make_video_loader,
)
from src.video_train import training_class_balance


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def inspect_epochs(run_dir: Path, manifest: dict):
    history = read_json(run_dir / "history.json")
    if not history or [row["epoch"] for row in history] != list(
        range(1, len(history) + 1)
    ):
        raise ValueError("Epoch sırası/logları eksik.")
    expected_best = max(history, key=lambda row: row["validation_metrics"]["macro_f1"])[
        "epoch"
    ]
    cache_audit = read_json(run_dir / "cache_audit.json")
    expected_paths = {
        r["path"] for r in cache_audit["splits"]["val"]["accepted_records"]
    }
    for row in history:
        epoch = row["epoch"]
        checkpoint = torch.load(
            run_dir / f"checkpoints/epoch_{epoch:03d}.pth",
            map_location="cpu",
            weights_only=True,
        )
        if (
            checkpoint["epoch"] != epoch
            or checkpoint["history"] != history[:epoch]
            or checkpoint["manifest_fingerprint"] != manifest["fingerprint"]
        ):
            raise ValueError("Epoch checkpoint/log/manifest uyuşmuyor.")
        logged = read_json(run_dir / f"epoch_{epoch:03d}_coverage.json")
        for name, coverage in (
            ("train", logged["train"]),
            ("val", logged["validation"]["coverage"]),
        ):
            counts = cache_audit["splits"][name]
            if (
                coverage["evaluated_videos"] != counts["accepted"]
                or coverage["attempted_videos"] != counts["total"]
                or len(coverage["rejected_videos"]) != counts["rejected"]
            ):
                raise ValueError("Epoch kabul/ret kapsamı cache ile uyuşmuyor.")
        predictions = logged["validation"]["predictions"]
        if {r["path"] for r in predictions} != expected_paths or len(
            predictions
        ) != len(expected_paths):
            raise ValueError("Epoch validation video kapsamı değişmiş.")
        metrics = metrics_from_rows(predictions)
        if (
            metrics != row["validation_metrics"]
            or logged["validation"]["video_loss"] != row["val_video_loss"]
        ):
            raise ValueError("Validation log/metrik uyuşmuyor.")
    best_path = run_dir / "checkpoints/best_video.pth"
    if file_sha256(best_path) != file_sha256(
        run_dir / f"checkpoints/epoch_{expected_best:03d}.pth"
    ):
        raise ValueError("Best checkpoint validation macro-F1 seçimiyle uyuşmuyor.")
    if file_sha256(run_dir / "checkpoints/last_video.pth") != file_sha256(
        run_dir / f"checkpoints/epoch_{len(history):03d}.pth"
    ):
        raise ValueError("Last checkpoint son epoch ile uyuşmuyor.")
    return history, expected_best


def metrics_from_rows(rows):
    return calculate_video_metrics(
        [r["target"] for r in rows],
        [r["prediction"] for r in rows],
        [r["fake_probability"] for r in rows],
    )


def family_distribution(records):
    families = defaultdict(list)
    for record in records:
        families[record["family"]].append(record)
    return dict(
        videos=len(records),
        folders=dict(Counter(r["path"].split("/")[0] for r in records)),
        unique_families=len(families),
        class_families={
            name: len({r["family"] for r in records if r["label"] == label})
            for label, name in enumerate(("REAL", "FAKE"))
        },
        mixed_families=sum(
            {r["label"] for r in rows} == {0, 1} for rows in families.values()
        ),
        family_size_histogram=dict(
            sorted(Counter(len(rows) for rows in families.values()).items())
        ),
        top_families=[
            dict(
                family=family,
                videos=len(rows),
                REAL=sum(r["label"] == 0 for r in rows),
                FAKE=sum(r["label"] == 1 for r in rows),
            )
            for family, rows in sorted(
                families.items(), key=lambda pair: (-len(pair[1]), pair[0])
            )[:10]
        ],
    )


def stratify_predictions(rows, items):
    lengths = {item["record"]["path"]: len(item["values"]) for item in items}
    by_length = {}
    for length in range(3, 7):
        subset = [r for r in rows if lengths[r["path"]] == length]
        by_length[length] = dict(
            n=len(subset),
            class_counts=dict(Counter(r["target"] for r in subset)),
            metrics=metrics_from_rows(subset)
            if {r["target"] for r in subset} == {0, 1}
            else None,
            metrics_limitation="Tek sınıflı/boş grupta binary karşılaştırma metrikleri raporlanmaz."
            if len({r["target"] for r in subset}) < 2
            else None,
        )
    by_folder = {}
    for folder in sorted({r["path"].split("/")[0] for r in rows}):
        subset = [r for r in rows if r["path"].startswith(folder + "/")]
        by_folder[folder] = dict(
            n=len(subset),
            correct=sum(r["prediction"] == r["target"] for r in subset),
            class_recall=sum(r["prediction"] == r["target"] for r in subset)
            / len(subset),
            mean_fake_probability=float(
                np.mean([r["fake_probability"] for r in subset])
            ),
        )
    return dict(by_valid_length=by_length, by_folder=by_folder)


@torch.no_grad()
def check_padding(model, items, device):
    selected = [
        next(item for item in items if len(item["values"]) == length)
        for length in range(3, 7)
    ]
    # Unsorted lengths, unequal padding, poisoned padding and individual-sequence reference.
    selected.reverse()
    lengths = torch.tensor([len(item["values"]) for item in selected])
    padded = torch.nn.utils.rnn.pad_sequence(
        [item["values"] for item in selected], batch_first=True
    ).to(device)
    reference = torch.cat(
        [
            model(
                item["values"].unsqueeze(0).to(device),
                torch.tensor([len(item["values"])]),
            )
            for item in selected
        ]
    )
    for index, length in enumerate(lengths):
        padded[index, length:] = float("nan")
    actual = model(padded, lengths)
    error = (actual - reference).abs().max().item()
    if not torch.isfinite(actual).all() or error > 1e-5:
        raise ValueError("Gerçek cache padding/masking kontrolü başarısız.")
    return dict(
        lengths=lengths.tolist(),
        poisoned_padding_ignored=True,
        max_absolute_logit_difference=error,
    )


def make_plots(history, output, best_epoch):
    epochs = [r["epoch"] for r in history]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
    axes[0].plot(
        epochs,
        [r["train_video_loss"] for r in history],
        "o-",
        label="Train weighted BCE (online)",
    )
    axes[0].plot(
        epochs,
        [r["val_video_loss"] for r in history],
        "o-",
        label="Validation unweighted BCE",
    )
    axes[0].set(ylabel="Video loss", title="Loss trends; objectives differ")
    axes[1].plot(
        epochs,
        [r["training_metrics"]["macro_f1"] for r in history],
        "o-",
        label="Train Macro-F1 (online)",
    )
    axes[1].plot(
        epochs,
        [r["validation_metrics"]["macro_f1"] for r in history],
        "o-",
        label="Validation Macro-F1",
    )
    axes[1].set(ylabel="Macro-F1", ylim=(0, 1), title="Video-level Macro-F1")
    for ax in axes:
        ax.axvline(
            best_epoch, linestyle="--", color="gray", label=f"Best epoch {best_epoch}"
        )
        ax.set(xlabel="Epoch", xticks=epochs)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    fig.savefig(output / "epoch_curves.png", dpi=180)
    fig.savefig(output / "epoch_curves.pdf")
    plt.close(fig)


def run_analysis(args):
    started = perf_counter()
    seed_everything(42)
    torch.set_num_threads(2)
    output = Path("outputs/video_v1/analyses") / uuid4().hex
    output.mkdir(parents=True, exist_ok=False)
    # Protect every photo output, current cache, original GRU checkpoint/log and experiments.
    protected = [
        p
        for p in Path("outputs").rglob("*")
        if p.is_file() and not p.is_relative_to(Path("outputs/video_v1"))
    ]
    protected += [
        p
        for directory in (args.run_dir, args.cache_dir)
        for p in directory.rglob("*")
        if p.is_file()
    ]
    protected += [
        args.manifest,
        Path("docs/EXPERIMENTS.md"),
        Path("src/model.py"),
        Path("src/train_v3.py"),
        Path("src/evaluate_v3.py"),
        Path("src/explainability.py"),
        Path("app.py"),
    ]
    snapshot = {str(path): file_sha256(path) for path in protected}
    (output / "preservation_before.json").write_text(
        json.dumps(snapshot, indent=2), encoding="utf-8"
    )
    manifest = load_manifest(args.manifest)
    history, best_epoch = inspect_epochs(args.run_dir, manifest)
    device = resolve_device(args.device)
    model, checkpoint = load_video_checkpoint(
        args.run_dir / "checkpoints/best_video.pth", device
    )
    settings = VideoSettings(**checkpoint["config"]["settings"])
    pipeline = FacePipeline(settings)
    if pipeline.signature != checkpoint["config"]["pipeline"]:
        raise ValueError("GRU pipeline imzası uyumsuz.")
    cache = FeatureCache(
        args.cache_dir, pipeline.signature, model.backbone_fingerprint()
    )
    cache_audit = audit_feature_cache(manifest, cache)
    if cache_audit["cache_fingerprint"] != checkpoint["config"]["cache_fingerprint"]:
        raise ValueError("Eğitimden sonra cache değişmiş.")
    balance = training_class_balance(cache_audit["splits"]["train"]["accepted_records"])
    if balance != checkpoint["config"]["training_class_balance"]:
        raise ValueError("Kabul edilmiş train sınıf ağırlıkları değişmiş.")
    items = {
        name: [
            cache.read(record, Path(manifest["dataset_root"]))
            for record in cache_audit["splits"][name]["accepted_records"]
        ]
        for name in ("train", "val")
    }
    means = {name: pooled_matrix(subset) for name, subset in items.items()}
    labels = np.array([item["record"]["label"] for item in items["train"]])
    fit_started = perf_counter()
    baseline = fit_baseline(means["train"], labels, balance["class_weights"])
    fit_seconds = perf_counter() - fit_started
    export_baseline(baseline, output / "mean_logistic_parameters.npz")
    baseline_results = {
        name: baseline_predictions(baseline, means[name], subset)
        for name, subset in items.items()
    }
    gru_results = {
        name: evaluate_video(
            model, make_video_loader(subset), device, cached=True, amp=False
        )
        for name, subset in items.items()
    }
    recorded = history[best_epoch - 1]["validation_metrics"]
    if any(
        abs(gru_results["val"]["metrics"][key] - recorded[key]) > 1e-9
        for key in ("accuracy", "macro_f1", "roc_auc")
    ):
        raise ValueError("Best GRU tekrar değerlendirme kayıtla uyuşmuyor.")
    padding = check_padding(model, items["val"], device)
    # Matched cohort: all originally 6-frame validation videos; change only prefix length.
    cohort = [item for item in items["val"] if len(item["values"]) == 6]
    prefix_results = {}
    for length in range(3, 7):
        truncated = [dict(item, values=item["values"][:length]) for item in cohort]
        gru = evaluate_video(
            model, make_video_loader(truncated), device, cached=True, amp=False
        )
        pooled = np.stack([temporal_mean(item["values"]) for item in truncated])
        prefix_results[length] = dict(
            n=len(cohort),
            GRU=gru["metrics"],
            mean_logistic=baseline_predictions(baseline, pooled, truncated)["metrics"],
        )
    reversed_items = [
        dict(item, values=item["values"].flip(0)) for item in items["val"]
    ]
    reversed_gru = evaluate_video(
        model, make_video_loader(reversed_items), device, cached=True, amp=False
    )
    make_plots(history, output, best_epoch)
    report = dict(
        schema="video_v1_gru_baseline_analysis_v1",
        source_run=str(args.run_dir.resolve()),
        best_epoch=best_epoch,
        epoch_logs=history,
        epoch_logs_verified=len(history),
        selection="validation_macro_f1",
        gru_best=dict(
            train=gru_results["train"]["metrics"], val=gru_results["val"]["metrics"]
        ),
        baseline=dict(
            model="temporal mean + train-only StandardScaler + weighted LogisticRegression",
            C=1.0,
            max_iter=1000,
            solver="lbfgs",
            iterations=baseline.named_steps["logisticregression"].n_iter_.tolist(),
            fits=1,
            validation_tuned=False,
            fit_seconds=fit_seconds,
            train=baseline_results["train"]["metrics"],
            val=baseline_results["val"]["metrics"],
        ),
        train_metric_scope="Both comparison train metrics use fixed final models; original epoch train metrics are online.",
        train_loss_scope="Epoch train BCE weighted; validation BCE unweighted, so absolute losses differ in objective.",
        accepted_counts={
            name: subset["accepted_counts"]
            for name, subset in cache_audit["splits"].items()
        },
        class_balance=balance,
        family_distribution={
            name: family_distribution(subset["accepted_records"])
            for name, subset in cache_audit["splits"].items()
        },
        cross_split_families=cache_audit["cross_split_families"],
        identity_overlap_train_val=len(
            {
                identity
                for item in items["train"]
                for identity in item["record"]["identities"]
            }
            & {
                identity
                for item in items["val"]
                for identity in item["record"]["identities"]
            }
        ),
        source_identity_verified=False,
        source_family_limit="Filename-derived carrier families; identity/donor relations unverified.",
        stratified={
            name: dict(
                GRU=stratify_predictions(gru_results[name]["predictions"], items[name]),
                mean_logistic=stratify_predictions(
                    baseline_results[name]["predictions"], items[name]
                ),
            )
            for name in ("train", "val")
        },
        padding_check=padding,
        matched_six_frame_cohort_prefix=prefix_results,
        reversed_validation=dict(
            n=len(reversed_items),
            metrics=reversed_gru["metrics"],
            interpretation="Diagnostic only; reversal preserves frames, changes temporal order. No tuning.",
        ),
        effective_optimizer=checkpoint["optimizer_state_dict"]["param_groups"][0],
        cache_fingerprint=cache_audit["cache_fingerprint"],
        official_test_accessed=False,
        gru_training_started=False,
        baseline_train_only=True,
    )
    for name in ("train", "val"):
        (output / f"{name}_predictions.json").write_text(
            json.dumps(
                dict(
                    GRU=gru_results[name]["predictions"],
                    mean_logistic=baseline_results[name]["predictions"],
                ),
                indent=2,
            ),
            encoding="utf-8",
        )
    changed = [
        path
        for path, sha in snapshot.items()
        if not Path(path).is_file() or file_sha256(Path(path)) != sha
    ]
    report["preservation"] = dict(files_checked=len(snapshot), changed=changed)
    if changed:
        raise RuntimeError("Korunan dosya değişmiş.")
    report["elapsed_seconds"] = perf_counter() - started
    (output / "analysis.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            dict(
                output=str(output.resolve()),
                best_epoch=best_epoch,
                GRU=report["gru_best"],
                baseline=report["baseline"],
                padding=padding,
                preservation=report["preservation"],
            ),
            ensure_ascii=False,
            indent=2,
        )
    )
    return output, report


def main():
    parser = argparse.ArgumentParser(
        description="Mevcut train/val cache; bir kısa LR baseline ve checkpoint analizi."
    )
    parser.add_argument("--analyze", action="store_true", required=True)
    parser.add_argument(
        "--run-dir", type=Path, default=Path("outputs/video_v1/gru_v1_fp32")
    )
    parser.add_argument(
        "--cache-dir", type=Path, default=Path("outputs/video_v1/cache")
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="auto")
    run_analysis(parser.parse_args())


if __name__ == "__main__":
    main()
