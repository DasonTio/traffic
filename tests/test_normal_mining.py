from pathlib import Path

import pytest

from scripts.evaluate_ground_truth import EventSpan
from scripts.mine_normal_sequences_from_gt import (
    FrameInterval,
    MiningSetupError,
    build_exclusion_intervals,
    frame_is_excluded,
    merge_intervals,
    validate_video_against_gt,
)


def _event(start_frame: int, end_frame: int) -> EventSpan:
    return EventSpan(
        event_id=f"gt_{start_frame}_{end_frame}",
        start_frame=start_frame,
        end_frame=end_frame,
        anomaly_type="lane_violation",
        class_name="Truck",
        lane_id="fast_lane_1",
        source="gt",
        raw={},
    )


def test_merge_intervals_combines_overlaps_and_adjacent_ranges():
    merged = merge_intervals(
        [
            FrameInterval(10, 20),
            FrameInterval(18, 25),
            FrameInterval(26, 30),
            FrameInterval(40, 45),
        ]
    )

    assert merged == [FrameInterval(10, 30), FrameInterval(40, 45)]


def test_build_exclusion_intervals_applies_buffer_and_window():
    intervals = build_exclusion_intervals(
        [_event(100, 120), _event(130, 150)],
        buffer_frames=10,
        frame_start=95,
        frame_end=155,
    )

    assert intervals == [FrameInterval(95, 155)]


def test_frame_is_excluded_advances_interval_pointer():
    intervals = [FrameInterval(20, 30), FrameInterval(50, 60)]
    excluded, index = frame_is_excluded(25, intervals, 0)
    assert excluded is True
    assert index == 0

    excluded, index = frame_is_excluded(40, intervals, index)
    assert excluded is False
    assert index == 1

    excluded, index = frame_is_excluded(55, intervals, index)
    assert excluded is True
    assert index == 1


def test_validate_video_against_gt_rejects_short_video():
    intervals = [FrameInterval(100, 140)]
    with pytest.raises(MiningSetupError, match="Use the annotated source video"):
        validate_video_against_gt(60, Path("dataset/ground_truth_events.csv"), intervals, ".video/video.mp4.mp4")
