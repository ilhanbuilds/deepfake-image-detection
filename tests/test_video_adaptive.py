"""Cascade routing, unique GPU work, short/decode failures, offline and pilot guards."""

import copy
from pathlib import Path
import sys

import numpy as np
import pytest

from scripts.benchmark_video_adaptive import benchmark, main, pilot_plan
from src.video_adaptive import (
    CHECKPOINT_SHA,
    AdaptiveVideo,
    authorize_input,
    method_summary,
    offline_rows,
    uncertain,
)
from src.video_forensic import summarize
from src.video_frame_study import nested_indices
from src.video_pipeline import sample_indices


class Capture:
    def __init__(self, cv):
        self.cv, self.index, self.closed = cv, -1, False

    def isOpened(self):
        return self.cv.opened

    def get(self, key):
        return self.cv.count

    def grab(self):
        assert not self.closed
        self.index += 1
        return self.index < self.cv.eof

    def retrieve(self):
        return (
            (False, None) if self.index in self.cv.bad_retrieve else (True, self.index)
        )

    def release(self):
        self.closed = True


class CV:
    CAP_PROP_FRAME_COUNT = 7

    def __init__(self, count=315, eof=None, bad_retrieve=(), opened=True):
        self.count, self.eof, self.bad_retrieve, self.opened = (
            count,
            count if eof is None else eof,
            bad_retrieve,
            opened,
        )
        self.captures = []

    def VideoCapture(self, source):
        cap = Capture(self)
        self.captures.append(cap)
        return cap


def engine(score=0.8, failures=(), cv=None, two_faces=False):
    calls = []

    def process(frame, index):
        assert frame == index
        calls.append(index)
        if index in failures:
            return dict(
                index=index,
                status="failed",
                reason="no_valid_face",
                processing_seconds=0.01,
            )
        faces = [dict(fake_probability=score)]
        if two_faces:
            faces.append(dict(fake_probability=score / 2))
        return dict(
            index=index,
            status="accepted",
            fake_probability=max(f["fake_probability"] for f in faces),
            faces=faces,
            processing_seconds=0.01,
        )

    cv = cv or CV()
    return AdaptiveVideo(cv, process), calls, cv


@pytest.mark.parametrize(
    "score,expected",
    [
        (0, False),
        (0.4, False),
        (0.4000001, True),
        (0.5, True),
        (0.5999999, True),
        (0.6, False),
        (1, False),
    ],
)
def test_fixed_strict_gate_boundaries(score, expected):
    assert uncertain(score) == expected


@pytest.mark.parametrize("score", [float("nan"), float("inf"), -0.01, 1.01])
def test_invalid_gate_probability_fails(score):
    with pytest.raises(ValueError):
        uncertain(score)


def test_confident_six_stops_and_cache_is_scoped_to_each_video_call():
    predictor, calls, cv = engine()
    row = predictor.predict(Path("one"))
    assert calls == sample_indices(315, 6)
    assert row["processed_frames"] == 6 and not row["expanded"]
    assert row["decode_passes"] == 1
    assert row["fake_probability"] == pytest.approx(0.8)
    predictor.predict(Path("two"))
    assert len(calls) == 12  # no score leakage between different videos
    assert all(cap.closed for cap in cv.captures)


def test_uncertain_only_processes_new_six_despite_second_decode():
    predictor, calls, cv = engine(0.5)
    row = predictor.predict(Path("one"))
    six = sample_indices(315, 6)
    twelve = nested_indices(315, six)[12]
    assert calls == six + sorted(set(twelve) - set(six))
    assert len(calls) == len(set(calls)) == row["processed_frames"] == 12
    assert row["decode_passes"] == 2 and row["decoded_grabs"] > 315
    assert row["expanded"] and row["expansion_reason"] == "margin"
    assert row["selected_indices"] == twelve
    assert row["frame_scores"] == [0.5] * 12 and row["prediction"] == 1
    assert all(cap.closed for cap in cv.captures)


def test_fixed_twelve_uses_one_fair_decode_pass():
    predictor, calls, _ = engine(0.5)
    row = predictor.predict(Path("one"), "12")
    assert len(calls) == len(set(calls)) == 12
    assert row["decode_passes"] == 1 and row["decoded_grabs"] == 315


def test_missing_initial_face_forces_expansion_without_retry_or_imputation():
    predictor, calls, _ = engine(0.9, failures={0})
    row = predictor.predict(Path("one"))
    assert row["expanded"] and row["expansion_reason"] == "incomplete_initial_six"
    assert calls.count(0) == 1 and len(calls) == 12
    assert row["status"] == "accepted" and row["valid_face_frames"] == 11
    assert row["fake_probability"] == pytest.approx(0.9)
    assert row["failures"] == [dict(index=0, reason="no_valid_face")]
    assert row["warnings"]


def test_all_no_face_rejects_without_fabricated_score():
    all_indices = nested_indices(315, sample_indices(315, 6))[12]
    predictor, calls, _ = engine(0.9, failures=set(all_indices))
    row = predictor.predict(Path("one"))
    assert (
        row["status"] == "rejected"
        and row["fake_probability"] is None
        and row["prediction"] is None
    )
    assert len(calls) == len(set(calls)) == 12


@pytest.mark.parametrize("count", [6, 7, 11])
def test_short_video_never_duplicates_available_frames(count):
    predictor, calls, _ = engine(0.5, cv=CV(count=count))
    row = predictor.predict(Path("one"))
    assert row["expanded"] and len(calls) == len(set(calls)) == count
    assert row["budget_shortfall"] == 12 - count and row["warnings"]


@pytest.mark.parametrize(
    "cv,reason", [(CV(count=5), "too_short"), (CV(opened=False), "video_open_failed")]
)
def test_unreadable_or_under_six_frames_reject_explicitly(cv, reason):
    predictor, calls, _ = engine(cv=cv)
    row = predictor.predict(Path("one"))
    assert row["status"] == "rejected" and reason in row["rejection_reason"]
    assert not calls and all(cap.closed for cap in cv.captures)


def test_early_eof_and_retrieve_failure_are_visible_and_not_retried():
    predictor, calls, _ = engine(0.5, cv=CV(count=315, eof=100, bad_retrieve={0}))
    row = predictor.predict(Path("one"))
    assert row["failures"] and len(calls) == len(set(calls))
    failed = [f["index"] for f in row["failures"]]
    assert len(failed) == len(set(failed))
    assert any(f["reason"] == "retrieve_failed" for f in row["failures"])
    assert any(f["reason"] == "decode_failed" for f in row["failures"])


def test_multiface_counts_faces_but_averages_frame_maxima():
    predictor, _, _ = engine(0.8, two_faces=True)
    row = predictor.predict(Path("one"))
    assert row["fake_probability"] == pytest.approx(0.8)
    assert row["classifier_face_calls"] == 12 and row["processed_frames"] == 6


def test_expected_frame_count_change_raises_and_closes_decoder():
    predictor, _, cv = engine()
    with pytest.raises(ValueError, match="count changed"):
        predictor.predict(Path("one"), expected_frame_count=316)
    assert all(cap.closed for cap in cv.captures)


def paired_fixture():
    sets = nested_indices(315, sample_indices(315, 6))
    rows = []
    for i in range(602):
        y = int(i >= 106)
        score = 0.9 if y else 0.1
        by_index = {n: score for n in sets[12]}
        if i == 0:  # REAL, uncertain wrong -> corrected
            by_index = {n: 0.55 if n in sets[6] else 0.1 for n in sets[12]}
        elif i == 106:  # FAKE, confident right -> fixed12 harmed, adaptive retains six
            by_index = {n: 0.7 if n in sets[6] else 0.1 for n in sets[12]}
        rows.append(
            dict(
                path=("Celeb-synthesis/" if y else "Celeb-real/") + str(i),
                target=y,
                indices_6=sets[6],
                indices_12=sets[12],
                frame_probabilities=by_index,
                score_6=float(np.mean([by_index[n] for n in sets[6]])),
                score_12=float(np.mean([by_index[n] for n in sets[12]])),
            )
        )
    methods = {}
    for mode in ("6", "12"):
        condition = [
            dict(
                path=r["path"],
                target=r["target"],
                status="accepted",
                fake_probability=r["score_" + mode],
                prediction=int(r["score_" + mode] >= 0.5),
            )
            for r in rows
        ]
        # Source helper expects a YouTube folder as well.
        condition[1]["path"] = rows[1]["path"] = "YouTube-real/1"
        methods[mode] = summarize(condition, len(rows))
    return dict(
        provenance=dict(epoch=6, checkpoint_sha256=CHECKPOINT_SHA),
        paired_rows=rows,
        methods=methods,
    )


def test_offline_fixed_replay_and_adaptive_mean_without_loading_model():
    paired = paired_fixture()
    rows = offline_rows(paired)
    assert rows["adaptive"][0]["expanded"]
    assert rows["adaptive"][0]["fake_probability"] == rows["12"][0]["fake_probability"]
    assert not rows["adaptive"][106]["expanded"]
    assert (
        rows["adaptive"][106]["fake_probability"] == rows["6"][106]["fake_probability"]
    )
    result = method_summary(rows["adaptive"])
    assert result["metrics"]["accuracy"] == 1
    assert result["expansion_rate"] == 1 / 602
    assert result["mean_processed_frames"] == (601 * 6 + 12) / 602
    assert result["latency"] is None and result["gpu_memory"] is None


@pytest.mark.parametrize(
    "mutation", ["missing", "duplicate", "bad_score", "mean", "wrong_checkpoint"]
)
def test_offline_invalid_or_wrong_model_scores_rejected(mutation):
    paired = paired_fixture()
    if mutation == "missing":
        paired["paired_rows"].pop()
    elif mutation == "duplicate":
        paired["paired_rows"][1] = copy.deepcopy(paired["paired_rows"][0])
    elif mutation == "bad_score":
        paired["paired_rows"][0]["frame_probabilities"][0] = float("nan")
    elif mutation == "mean":
        paired["paired_rows"][0]["score_6"] += 0.01
    else:
        paired["provenance"]["checkpoint_sha256"] = "other"
    with pytest.raises(ValueError):
        offline_rows(paired)


def test_pilot_preselection_uses_paths_not_scores_and_is_order_independent():
    records = [
        dict(path=source + f"/{i}.mp4")
        for source in ("Celeb-real", "YouTube-real", "Celeb-synthesis")
        for i in range(10)
    ]
    plan = pilot_plan(records)
    assert plan == pilot_plan(list(reversed(records)))
    assert len(plan["records"]) == 9
    assert all(
        sum(r["path"].startswith(source + "/") for r in plan["records"]) == 3
        for source in ("Celeb-real", "YouTube-real", "Celeb-synthesis")
    )


def test_official_test_and_quarantine_paths_rejected_before_decode(tmp_path):
    root = tmp_path / "dataset"
    record = dict(path="Celeb-real/train.mp4")
    for path in (root / "Celeb-real/test.mp4", root / "Celeb-synthesis/quarantine.mp4"):
        with pytest.raises(ValueError, match="locked"):
            authorize_input(path, root, [record])
    assert authorize_input(root / record["path"], root, [record]) == record
    assert authorize_input(tmp_path / "new_user_video.mp4", root, [record]) is None


def test_cli_requires_explicit_mode_before_io(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["benchmark_video_adaptive"])
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2


def test_benchmark_counts_unique_videos_for_accuracy_and_repeats_only_for_latency(
    tmp_path, monkeypatch
):
    import scripts.benchmark_video_adaptive as driver
    from src.video_forensic import read_json, write_json

    root, out = tmp_path / "inputs", tmp_path / "report"
    out.mkdir()
    records = []
    for source in ("Celeb-real", "YouTube-real", "Celeb-synthesis"):
        for i in range(3):
            record = dict(
                path=source + f"/{i}.mp4", label=int(source == "Celeb-synthesis")
            )
            source_path = root / record["path"]
            source_path.parent.mkdir(parents=True, exist_ok=True)
            source_path.write_bytes(b"synthetic decoder fixture")
            records.append(record)
    plan_path = tmp_path / "plan.json"
    write_json(plan_path, pilot_plan(records))
    monkeypatch.setattr(
        driver,
        "verified_source",
        lambda root, record, audit: (root / record["path"], 315),
    )
    monkeypatch.setattr(driver.torch.cuda, "synchronize", lambda: None)
    predictor, _, _ = engine(0.8)
    benchmark(out, predictor, root, {}, records, plan_path, 3)
    result = read_json(out / "benchmark.json")
    assert len(result["rows"]) == 81
    for mode in ("6", "12", "adaptive"):
        summary = result["methods"][mode]
        assert summary["attempted"] == summary["independent_videos"] == 9
        assert summary["measured_runs"] == summary["latency"]["videos"] == 27
        assert summary["metrics"]["per_class"]["REAL"]["support"] == 6
        assert summary["metrics"]["per_class"]["FAKE"]["support"] == 3
        assert summary["false_alarms"] == 6
    for index in range(9):
        first_modes = [
            result["rows"][repeat * 27 + index * 3]["mode"] for repeat in range(3)
        ]
        assert set(first_modes) == {"6", "12", "adaptive"}
