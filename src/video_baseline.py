"""Train-only standardized temporal mean + one weighted Logistic Regression fit."""

import warnings

import numpy as np
import torch
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.video_evaluate import calculate_video_metrics


def temporal_mean(features: torch.Tensor, length: int | None = None) -> np.ndarray:
    if features.ndim != 2 or features.shape[1] != 1280:
        raise ValueError("Temporal mean girdisi [T,1280] olmalı.")
    length = len(features) if length is None else length
    if not 3 <= length <= min(6, len(features)):
        raise ValueError("Mean pooling için 3–6 geçerli kare gerekli.")
    valid = features[:length].float()
    if not torch.isfinite(valid).all():
        raise ValueError("Mean pooling feature NaN/Inf içeriyor.")
    return valid.mean(dim=0).cpu().numpy()


def pooled_matrix(items: list[dict]) -> np.ndarray:
    if not items or any(item["values"] is None for item in items):
        raise ValueError("Baseline yalnız kabul edilmiş cache kullanır.")
    return np.stack([temporal_mean(item["values"]) for item in items])


def fit_baseline(features: np.ndarray, labels: np.ndarray, class_weights: dict):
    if (
        features.ndim != 2
        or features.shape[1] != 1280
        or not np.isfinite(features).all()
    ):
        raise ValueError("Baseline train özellikleri geçersiz.")
    if labels.shape != (len(features),) or set(labels.tolist()) != {0, 1}:
        raise ValueError("Baseline train iki video sınıfını içermeli.")
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=1.0,
            solver="lbfgs",
            max_iter=1000,
            random_state=42,
            class_weight={0: class_weights["REAL"], 1: class_weights["FAKE"]},
        ),
    )
    # No validation fitting, parameter sweep, threshold tuning, or silent nonconvergence.
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        model.fit(features, labels)
    return model


def baseline_predictions(model, features: np.ndarray, items: list[dict]) -> dict:
    scores = model.predict_proba(features)[:, 1].tolist()
    targets = [item["record"]["label"] for item in items]
    predictions = [int(score >= 0.5) for score in scores]
    return dict(
        metrics=calculate_video_metrics(targets, predictions, scores),
        predictions=[
            dict(
                path=item["record"]["path"],
                target=target,
                fake_probability=score,
                prediction=prediction,
            )
            for item, target, score, prediction in zip(
                items, targets, scores, predictions, strict=True
            )
        ],
    )


def export_baseline(model, path):
    scaler, classifier = (
        model.named_steps["standardscaler"],
        model.named_steps["logisticregression"],
    )
    with path.open("xb") as handle:
        np.savez_compressed(
            handle,
            mean=scaler.mean_,
            scale=scaler.scale_,
            coefficient=classifier.coef_,
            intercept=classifier.intercept_,
            classes=classifier.classes_,
        )
