from pathlib import Path

import pytest

from scripts.evaluate_ground_truth import evaluate_run, write_evaluation_outputs


def _write_csv(path: Path, header: list[str], rows: list[list[object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [",".join(header)]
    for row in rows:
        lines.append(",".join(str(cell) for cell in row))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_evaluate_run_computes_metrics_and_writes_outputs(tmp_path):
    run_dir = tmp_path / "runs" / "test_run"
    gt_path = tmp_path / "dataset" / "ground_truth_events.csv"
    events_path = run_dir / "events.csv"

    _write_csv(
        gt_path,
        [
            "event_gt_id", "source_video", "chunk_start_frame", "chunk_end_frame", "start_frame", "end_frame",
            "start_time_s", "end_time_s", "anomaly_type", "actual_class", "lane_id", "notes",
        ],
        [
            ["gt_001", "video.mp4", 1, 300, 100, 150, 0.0, 0.0, "lane_violation", "Truck", "fast_lane_1", ""],
        ],
    )
    _write_csv(
        events_path,
        [
            "event_id", "camera_id", "track_id", "class_name", "anomaly_type", "severity", "start_frame", "end_frame",
            "rule_score", "ganomaly_score", "fused_score", "lane_id", "explanation", "crop_path", "frame_path",
        ],
        [
            ["pred_001", "cam", 7, "Truck", "lane_violation", "warning", 110, 140, 0.8, 0.0, 0.8, "fast_lane_1", "match", "", ""],
            ["pred_002", "cam", 8, "Truck", "wrong_way", "warning", 200, 220, 0.7, 0.0, 0.7, "fast_lane_2", "fp", "", ""],
        ],
    )

    summary = evaluate_run(run_dir, gt_path)

    assert summary.total_gt == 1
    assert summary.total_pred == 2
    assert len(summary.matches) == 1
    assert len(summary.false_negatives) == 0
    assert len(summary.false_positives) == 1
    assert summary.precision == pytest.approx(0.5)
    assert summary.recall == pytest.approx(1.0)
    assert summary.f1 == pytest.approx(2 * 0.5 * 1.0 / 1.5)
    assert summary.lane_agreement == pytest.approx(1.0)
    assert summary.class_agreement == pytest.approx(1.0)

    eval_dir = write_evaluation_outputs(summary)
    assert (eval_dir / "matched_events.csv").exists()
    assert (eval_dir / "false_negatives.csv").exists()
    assert (eval_dir / "false_positives.csv").exists()
    assert (eval_dir / "ground_truth_report.md").exists()
