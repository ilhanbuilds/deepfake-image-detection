"""Label-free, bounded, within-video quality weighting of fixed frame scores.

This is a project hypothesis, not a claim of global novelty or measured gain.
All constants are fixed before inspecting the validation comparison.
"""

import argparse
from pathlib import Path

import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

from src.video_forensic import accepted_roster, read_json, summarize, write_json


def relative_quality(values):
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("Finite nonempty quality vector required.")
    if len(values) == 1 or np.ptp(values) < 1e-12:
        return np.full(len(values), 0.5)
    return (rankdata(values, method="average") - 1) / (len(values) - 1)


def quality_components(report):
    sharpness, retention = [], []
    for frame in report["frames"]:
        if not frame["faces"]:
            raise ValueError("A scored frame must contain a detected face.")
        # Minimum quality across retained faces; never choose quality using fake score.
        sharpness.append(min(f["sharpness"] for f in frame["faces"]))
        ratios = []
        for face in frame["faces"]:
            x0, y0, x1, y1 = face["bbox"]
            h, w = face["crop_size"]
            expected = (1.25 * (x1 - x0) + 1) * (1.25 * (y1 - y0) + 1)
            if expected <= 0:
                raise ValueError("Invalid bbox geometry.")
            ratios.append(float(np.clip(h * w / expected, 0, 1)))
        retention.append(min(ratios))
    return np.asarray(sharpness), np.asarray(retention)


def fusion_weights(report, mode="full", mixing=0.5):
    if mode not in ("mean", "full", "sharpness", "boundary") or not 0 <= mixing <= 1:
        raise ValueError("Invalid fusion setting.")
    sharpness, retention = quality_components(report)
    n = len(sharpness)
    if not n:
        raise ValueError("No valid frames; no fabricated prediction.")
    if mode == "mean":
        return np.full(n, 1 / n)
    sharpness = relative_quality(sharpness)
    q = (
        (sharpness + retention) / 2
        if mode == "full"
        else (sharpness if mode == "sharpness" else retention)
    )
    reliability = 0.5 + q
    return (1 - mixing) / n + mixing * reliability / reliability.sum()


def fuse(scores, report, mode="full"):
    scores = np.asarray(scores, dtype=float)
    weights = fusion_weights(report, mode)
    if (
        scores.shape != weights.shape
        or not np.isfinite(scores).all()
        or np.any((scores < 0) | (scores > 1))
    ):
        raise ValueError("One finite probability per valid frame required.")
    return float(scores @ weights), weights.tolist()


def compare(evaluation):
    comparisons = {}
    for mode in ("mean", "full", "sharpness", "boundary"):
        rows = []
        for original in evaluation["rows"]:
            row = dict(original)
            if row["status"] == "accepted":
                score, weights = fuse(row["frame_scores"], row["report"], mode)
                row.update(
                    fake_probability=score,
                    prediction=int(score >= 0.5),
                    fusion_weights=weights,
                )
            rows.append(row)
        comparisons[mode] = dict(**summarize(rows, len(rows)), rows=rows)
    valid = [r for r in evaluation["rows"] if r["status"] == "accepted"]
    means = [float(quality_components(r["report"])[0].mean()) for r in valid]
    real = [r for r in valid if r["target"] == 0]
    real_quality = [float(quality_components(r["report"])[0].mean()) for r in real]
    source_labels = [int(r["path"].startswith("YouTube-real/")) for r in real]
    bias = dict(
        description="Diagnostic AUC of raw mean sharpness; NOT a trained detector, not used for weighting calibration.",
        label_auc=float(roc_auc_score([r["target"] for r in valid], means))
        if len({r["target"] for r in valid}) == 2
        else None,
        real_source_auc=float(roc_auc_score(source_labels, real_quality))
        if len(set(source_labels)) == 2
        else None,
    )
    return dict(
        provenance=evaluation["provenance"],
        methods=comparisons,
        quality_source_diagnostic=bias,
        constants_selected_on_validation=False,
        inference_deterministic=True,
        limitations="Exploratory validation only; no claimed robustness benefit until corruption experiments.",
    )


def paired_family_bootstrap(comparison, records, repeats=1000):
    """Conditional validation uncertainty; known clip families are the sampling unit.

    Not a new holdout, and not a claim of independent real-person identities.
    """
    baseline = [
        r for r in comparison["methods"]["mean"]["rows"] if r["status"] == "accepted"
    ]
    proposed = {
        r["path"]: r
        for r in comparison["methods"]["full"]["rows"]
        if r["status"] == "accepted"
    }
    families = {r["path"]: r["family"] for r in records}
    groups = {}
    for index, row in enumerate(baseline):
        groups.setdefault(families[row["path"]], []).append(index)
    indices = [np.asarray(v) for v in groups.values()]
    y = np.asarray([r["target"] for r in baseline])
    a = np.asarray([r["fake_probability"] for r in baseline])
    b = np.asarray([proposed[r["path"]]["fake_probability"] for r in baseline])

    def macro(labels, scores):
        predictions = scores >= 0.5
        values = []
        for label in (0, 1):
            tp = np.sum((labels == label) & (predictions == label))
            fp = np.sum((labels != label) & (predictions == label))
            fn = np.sum((labels == label) & (predictions != label))
            values.append(2 * tp / max(1, 2 * tp + fp + fn))
        return float(np.mean(values))

    result = {}
    for seed in (42, 123, 2026):
        rng = np.random.default_rng(seed)
        deltas = []
        for _ in range(repeats):
            sample = np.concatenate(
                [indices[i] for i in rng.integers(len(indices), size=len(indices))]
            )
            target = y[sample]
            if len(set(target.tolist())) < 2:
                continue
            deltas.append(
                (
                    roc_auc_score(target, b[sample]) - roc_auc_score(target, a[sample]),
                    macro(target, b[sample]) - macro(target, a[sample]),
                )
            )
        array = np.asarray(deltas)
        result[str(seed)] = dict(
            valid_draws=len(deltas),
            roc_auc_delta_ci95=np.quantile(array[:, 0], [0.025, 0.975]).tolist(),
            macro_f1_delta_ci95=np.quantile(array[:, 1], [0.025, 0.975]).tolist(),
        )
    return dict(
        family_count=len(groups),
        repeats_per_seed=repeats,
        seeds=result,
        limitation="Known filename families only; residual shared identities/sources and validation selection bias remain. Bootstrap seeds are not training seeds.",
    )


def main():
    parser = argparse.ArgumentParser(
        description="Paired fusion and ablations; reuse identical checkpoint/frame scores."
    )
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Existing comparison protected.")
    result = compare(read_json(args.evaluation))
    if result["provenance"].get("split") == "accepted_validation":
        _, _, records = accepted_roster("val")
        result["paired_family_bootstrap"] = paired_family_bootstrap(result, records)
    write_json(args.output, result)
    print(
        {
            m: {k: r["metrics"][k] for k in ("roc_auc", "macro_f1")}
            for m, r in result["methods"].items()
        }
    )


if __name__ == "__main__":
    main()
