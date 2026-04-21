from pathlib import Path

import pytest

from scripts.evaluate_ground_truth import EvaluationSummary
from scripts.test_model import (
    EvaluationSetupError,
    build_appearance_test_report,
    build_appearance_test_result,
    build_test_report,
    build_test_result,
    get_run_frame_range,
    validate_run_against_ground_truth,
)
from traffic_anomaly.appearance_eval import AppearanceEvaluationSummary, AppearanceMetrics


def _summary(*, precision: float, recall: float, f1: float) -> EvaluationSummary:
    return EvaluationSummary(
        run_dir=Path("/tmp/run"),
        gt_path=Path("/tmp/gt.csv"),
        events_path=Path("/tmp/run/events.csv"),
        total_gt=10,
        total_pred=9,
        matches=[],
        false_negatives=[],
        false_positives=[],
        precision=precision,
        recall=recall,
        f1=f1,
        lane_agreement=0.9,
        class_agreement=0.8,
    )


def _appearance_summary(*, precision: float, recall: float, f1: float) -> AppearanceEvaluationSummary:
    metrics = AppearanceMetrics(
        total=10,
        positives=4,
        negatives=6,
        true_positives=4,
        false_positives=1,
        false_negatives=0,
        true_negatives=5,
        precision=precision,
        recall=recall,
        f1=f1,
        auroc=0.91,
        auprc=0.88,
    )
    return AppearanceEvaluationSummary(
        model_name="ganomaly",
        gt_path=Path("/tmp/appearance_gt.csv"),
        threshold=1.0,
        total_samples=10,
        metrics=metrics,
        predictions=[],
        per_group={"car": metrics},
    )


def test_build_test_result_passes_when_all_thresholds_are_met():
    summary = _summary(precision=0.85, recall=0.80, f1=0.82)
    result = build_test_result(summary, min_precision=0.80, min_recall=0.70, min_f1=0.75)

    assert result["passed"] is True
    assert result["verdict"] == "PASS"
    assert result["checks"] == {"precision": True, "recall": True, "f1": True}


def test_build_test_report_includes_verdict_and_thresholds():
    summary = _summary(precision=0.60, recall=0.55, f1=0.57)
    result = build_test_result(summary, min_precision=0.80, min_recall=0.70, min_f1=0.75)
    report = build_test_report(summary, result)

    assert result["passed"] is False
    assert "Verdict: **FAIL**" in report
    assert "Precision >= 80.0%: FAIL" in report
    assert "Recall >= 70.0%: FAIL" in report
    assert "F1 >= 75.0%: FAIL" in report


def test_build_appearance_test_result_and_report():
    summary = _appearance_summary(precision=0.82, recall=0.79, f1=0.80)
    result = build_appearance_test_result(summary, min_precision=0.80, min_recall=0.70, min_f1=0.75)
    report = build_appearance_test_report(summary, result)

    assert result["passed"] is True
    assert result["verdict"] == "PASS"
    assert "Appearance Model Test Report" in report
    assert "AUROC: 0.910" in report


def test_get_run_frame_range_prefers_run_metadata(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "run_metadata.json").write_text(
        '{"processed_frame_min": 1, "processed_frame_max": 60}',
        encoding="utf-8",
    )

    assert get_run_frame_range(run_dir) == (1, 60)


def test_validate_run_against_ground_truth_rejects_non_overlapping_ranges(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "run_metadata.json").write_text(
        '{"processed_frame_min": 1, "processed_frame_max": 60}',
        encoding="utf-8",
    )
    (run_dir / "events.csv").write_text("event_id,camera_id,track_id,class_name,anomaly_type,severity,start_frame,end_frame,rule_score,ganomaly_score,fused_score,lane_id,explanation,crop_path,frame_path\n", encoding="utf-8")
    gt_path = tmp_path / "gt.csv"
    gt_path.write_text(
        "event_gt_id,source_video,chunk_start_frame,chunk_end_frame,start_frame,end_frame,start_time_s,end_time_s,anomaly_type,actual_class,lane_id,notes\n"
        "gt_001,video.mp4,1,10000,3799,3857,0,0,lane_violation,Truck,fast_lane_2,\n",
        encoding="utf-8",
    )

    with pytest.raises(EvaluationSetupError, match="No frame overlap"):
        validate_run_against_ground_truth(run_dir, gt_path)
