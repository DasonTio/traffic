import csv

from traffic_anomaly.review_ganomaly import (
    STATUS_APPROVED,
    STATUS_PENDING,
    review_indices,
    save_manifest_rows,
)


def test_review_indices_pending_only():
    rows = [
        {"sample_id": "1", "status": STATUS_PENDING},
        {"sample_id": "2", "status": STATUS_APPROVED},
        {"sample_id": "3", "status": STATUS_PENDING},
    ]
    assert review_indices(rows, pending_only=True) == [0, 2]
    assert review_indices(rows, pending_only=False) == [0, 1, 2]


def test_save_manifest_rows_roundtrip(tmp_path):
    manifest_path = tmp_path / "review_manifest.csv"
    fieldnames = ["sample_id", "status"]
    rows = [
        {"sample_id": "1", "status": STATUS_PENDING},
        {"sample_id": "2", "status": STATUS_APPROVED},
    ]

    save_manifest_rows(manifest_path, fieldnames, rows)

    with manifest_path.open("r", newline="", encoding="utf-8") as handle:
        loaded = list(csv.DictReader(handle))
    assert loaded == rows
