"""Nested sampling, paired aggregation, cache integrity and fresh-baseline guards."""

import copy
import hashlib

import numpy as np
import pytest
import torch

from scripts.recheck_video_frame_timing import compare_frame
from scripts.report_video_frames import (
    macro_f1,
    paired_uncertainty,
    timing_sensitivity,
    transitions,
)
from src import video_frame_study
from src.video_frame_study import (
    BUDGETS,
    compare_baseline,
    condition_row,
    decode_selected,
    nested_indices,
    pilot_records,
    save_cache,
    validate_cache,
)
from src.video_pipeline import sample_indices


@pytest.mark.parametrize("count", [6, 7, 12, 25, 32, 33, 315, 1000])
def test_nested_sampling_preserves_six_and_never_duplicates(count):
    original = sample_indices(count)
    sets = nested_indices(count, original)
    previous = set()
    for budget in BUDGETS:
        indices = sets[budget]
        assert len(indices) == min(budget, count)
        assert indices == sorted(set(indices))
        assert previous <= set(indices)
        assert indices[0] == 0 and indices[-1] == count - 1
        previous = set(indices)
        gaps = np.diff(indices)
        assert max(gaps) <= 2 * min(gaps) + 1
    assert sets[6] == original
    assert sets == nested_indices(count, original)


def test_historical_example_and_gap_tie_rule():
    sets = nested_indices(315, [0, 63, 126, 188, 251, 314])
    assert sets[12] == [0, 31, 47, 63, 94, 126, 157, 188, 219, 251, 282, 314]


def test_changed_original_indices_are_rejected():
    with pytest.raises(ValueError, match="Historical"):
        nested_indices(315, [0, 62, 125, 188, 251, 314])


def cache():
    original = sample_indices(40)
    return dict(
        record=dict(path="Celeb-real/example.mp4", label=0),
        signature=dict(version=1),
        frame_count=40,
        original_indices=original,
        sets={str(k): v for k, v in nested_indices(40, original).items()},
        expanded=False,
        frames=[
            dict(
                index=i,
                status="accepted",
                fake_probability=p,
                faces=[dict(fake_probability=p)],
                processing_seconds=0.1,
            )
            for i, p in zip(original, [0.1, 0.2, 0.4, 0.6, 0.8, 0.9], strict=True)
        ],
        passes=[dict(decode_seconds=1.0)],
    )


def test_condition_uses_frame_mean_fixed_threshold_and_explicit_failures():
    c = cache()
    row = condition_row(c, 6)
    assert row["fake_probability"] == pytest.approx(0.5)
    assert row["prediction"] == 1
    assert row["attributed_seconds"] == pytest.approx(1.6)
    c["frames"][0] = dict(
        index=0, status="failed", reason="no_valid_face", processing_seconds=0.1
    )
    row = condition_row(c, 6)
    assert row["valid_face_frames"] == 5
    assert len(row["failed_frames"]) == 1
    assert row["fake_probability"] == pytest.approx(2.9 / 5)
    with pytest.raises(ValueError, match="not yet computed"):
        condition_row(c, 12)


def test_cache_rejects_tampering_provenance_and_duplicate_frames(tmp_path):
    c = cache()
    save_cache(tmp_path / "cache.json", c)
    validate_cache(c, c["record"], c["signature"])
    changed = copy.deepcopy(c)
    changed["frames"][0]["fake_probability"] = 0.3
    with pytest.raises(ValueError, match="integrity"):
        validate_cache(changed, c["record"], c["signature"])
    with pytest.raises(ValueError, match="provenance"):
        validate_cache(c, c["record"], dict(version=2))
    c["frames"].append(c["frames"][0])
    save_cache(tmp_path / "cache.json", c)
    with pytest.raises(ValueError, match="duplicate"):
        validate_cache(c, c["record"], c["signature"])


def test_multiface_max_integrity_guard(tmp_path):
    c = cache()
    c["frames"][0]["faces"].append(dict(fake_probability=0.95))
    save_cache(tmp_path / "cache.json", c)
    with pytest.raises(ValueError, match="multiface"):
        validate_cache(c, c["record"], c["signature"])


def test_pilot_has_27_videos_all_sources_and_order_independence():
    records = [
        dict(path=f"{source}/{i}.mp4")
        for source in ("Celeb-real", "YouTube-real", "Celeb-synthesis")
        for i in range(20)
    ]
    chosen = pilot_records(records)
    assert len(chosen) == 27
    assert chosen == pilot_records(list(reversed(records)))
    assert all(
        sum(r["path"].startswith(s + "/") for r in chosen) == 9
        for s in ("Celeb-real", "YouTube-real", "Celeb-synthesis")
    )


def test_baseline_detects_crop_or_score_drift():
    c = cache()
    pixels = torch.zeros((6, 3, 380, 380), dtype=torch.uint8)
    report = dict(status="accepted", selected_indices=c["original_indices"], frames=[])
    for n, frame in enumerate(c["frames"]):
        frame["faces"][0].update(
            bbox=[1, 2, 3, 4],
            crop_size=[3, 3],
            crop_sha256=hashlib.sha256(pixels[n].numpy().tobytes()).hexdigest(),
        )
        report["frames"].append(
            dict(
                index=frame["index"],
                faces=[dict(crop_index=n, bbox=[1, 2, 3, 4], crop_size=[3, 3])],
            )
        )
    historical = dict(
        report=report, face_scores=[f["fake_probability"] for f in c["frames"]]
    )
    assert compare_baseline(c, historical, pixels, report)["crop_pixels_equal"]
    c["frames"][0]["faces"][0]["fake_probability"] += 0.01
    with pytest.raises(ValueError, match="score drift"):
        compare_baseline(c, historical, pixels, report)
    c["frames"][0]["faces"][0]["crop_sha256"] = "changed"
    with pytest.raises(ValueError, match="crop pixels"):
        compare_baseline(c, historical, pixels, report)


def test_sequential_decoder_retrieves_only_requested_frames_and_reports_eof(
    monkeypatch,
):
    class Capture:
        def __init__(self):
            self.position = -1
            self.retrieved = []
            self.released = False

        def isOpened(self):
            return True

        def get(self, prop):
            return 40

        def grab(self):
            self.position += 1
            return self.position < 20

        def retrieve(self):
            self.retrieved.append(self.position)
            return self.position != 10, self.position

        def release(self):
            self.released = True

    capture = Capture()

    class CV:
        CAP_PROP_FRAME_COUNT = 7

        def VideoCapture(self, source):
            return capture

    class Extractor:
        cv = CV()

    monkeypatch.setattr(
        video_frame_study,
        "process_frame",
        lambda extractor, model, frame, index: dict(index=index, status="accepted"),
    )
    frames, timing = decode_selected(
        "synthetic", [1, 10, 25, 39], 40, Extractor(), None
    )
    assert capture.released and capture.retrieved == [1, 10]
    assert [f["index"] for f in frames] == [1, 10, 25, 39]
    assert [f["reason"] for f in frames[1:]] == ["decode_failed"] * 3
    assert timing["decode_seconds"] > 0


def test_macro_and_paired_analysis_identical_scores_have_zero_delta():
    records = [dict(path=f"example/{i}", family=f"family{i // 2}") for i in range(8)]
    rows = [
        dict(
            path=r["path"],
            target=i % 2,
            prediction=i % 2,
            status="accepted",
            fake_probability=0.9 if i % 2 else 0.1,
        )
        for i, r in enumerate(records)
    ]
    methods = {str(n): dict(rows=copy.deepcopy(rows)) for n in BUDGETS}
    assert macro_f1(np.array([0, 1]), np.array([0.1, 0.9])) == 1
    result = paired_uncertainty(methods, records, repeats=10)
    assert result["family_count"] == 4
    for seed in result["seeds"].values():
        for delta in seed.values():
            assert delta["roc_auc_delta_ci95"] == [0, 0]
            assert delta["macro_f1_delta_ci95"] == [0, 0]
    methods["12"]["rows"][0].update(prediction=1, fake_probability=0.8)
    changes = transitions(methods)
    assert len(changes["12"]["harmed"]) == 1
    assert not changes["12"]["corrected"]
    assert changes["12"]["margin_bins"][-1]["harmed"] == 1
    methods["24"]["rows"].reverse()
    with pytest.raises(ValueError, match="identical ordered"):
        paired_uncertainty(methods, records, repeats=10)


def test_timing_sensitivity_keeps_every_video_and_preserves_raw_measurements():
    methods = {
        str(n): dict(
            rows=[
                dict(
                    path="a",
                    selected_indices=[0, 177] if n == 32 else [0],
                    attributed_seconds=10.0,
                ),
                dict(path="b", selected_indices=[0], attributed_seconds=4.0),
            ],
            attributed_seconds=14.0,
            mean_seconds=7.0,
        )
        for n in BUDGETS
    }
    original = copy.deepcopy(methods)
    recheck = dict(
        frames=[
            dict(
                path="a",
                index=177,
                raw_processing_seconds=6.0,
                median_remeasured_seconds=0.5,
                affected_budgets=[32],
            )
        ]
    )
    result = timing_sensitivity(methods, recheck)
    assert methods == original
    assert result["6"]["remeasured_mean_seconds_estimate"] == 7.0
    assert result["32"]["raw_mean_seconds"] == 7.0
    assert result["32"]["remeasured_mean_seconds_estimate"] == 4.25
    recheck["frames"].append(recheck["frames"][0])
    with pytest.raises(ValueError, match="Duplicate"):
        timing_sensitivity(methods, recheck)


def test_timing_recheck_rejects_changed_predictions_or_crops():
    frame = dict(
        status="accepted",
        faces=[
            dict(
                bbox=[0, 0, 2, 2],
                crop_size=[3, 3],
                crop_sha256="pixels",
                fake_probability=0.6,
            )
        ],
    )
    repeated = copy.deepcopy(frame)
    assert compare_frame(frame, repeated) == 0
    repeated["faces"][0]["fake_probability"] = 0.61
    with pytest.raises(ValueError, match="scores"):
        compare_frame(frame, repeated)
    repeated = copy.deepcopy(frame)
    repeated["faces"][0]["crop_sha256"] = "changed"
    with pytest.raises(ValueError, match="pixels"):
        compare_frame(frame, repeated)
