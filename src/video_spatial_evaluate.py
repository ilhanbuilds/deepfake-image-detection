"""Overall metrics, per-source recall and explicit two-class source-pair AUC."""

from sklearn.metrics import roc_auc_score

from src.video_evaluate import calculate_video_metrics
from src.video_sampling import comparison_metrics, source_metrics


def spatial_source_metrics(rows):
    result = source_metrics(rows)
    fakes = [r for r in rows if r["path"].startswith("Celeb-synthesis/")]
    for source in ("Celeb-real", "YouTube-real"):
        real = [r for r in rows if r["path"].startswith(source + "/")]
        pair = real + fakes
        result[source]["roc_auc_vs_celeb_synthesis"] = (
            float(
                roc_auc_score(
                    [r["target"] for r in pair], [r["fake_probability"] for r in pair]
                )
            )
            if real and fakes
            else None
        )
    result["Celeb-synthesis"]["single_source_roc_auc"] = None
    result["Celeb-synthesis"]["auc_reason"] = (
        "Single-class folder; AUC requires a REAL comparison group."
    )
    return result


def summarize_evaluation(result):
    return dict(
        **comparison_metrics(result),
        source_pairs=spatial_source_metrics(result["predictions"]),
    )


def validate_recorded_predictions(result, records):
    rows = result["predictions"]
    expected = {r["path"]: r["label"] for r in records}
    if len(rows) != len(expected) or {r["path"]: r["target"] for r in rows} != expected:
        raise ValueError("Fixed değerlendirme roster/etiket/kapsam uyuşmuyor.")
    if any(r["prediction"] != int(r["fake_probability"] >= 0.5) for r in rows):
        raise ValueError("Inference threshold 0.5 uyuşmuyor.")
    actual = calculate_video_metrics(
        [r["target"] for r in rows],
        [r["prediction"] for r in rows],
        [r["fake_probability"] for r in rows],
    )
    if actual != result["metrics"]:
        raise ValueError("Kaydedilmiş metrikler tahminlerden yeniden üretilemiyor.")
    return summarize_evaluation(result)
