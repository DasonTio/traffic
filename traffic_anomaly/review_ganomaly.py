from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np


STATUS_APPROVED = "approved"
STATUS_REJECTED = "rejected"
STATUS_PENDING = "pending"


def load_manifest_rows(manifest_path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with manifest_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])
    if not fieldnames:
        raise RuntimeError(f"Manifest has no header: {manifest_path}")
    return rows, fieldnames


def save_manifest_rows(manifest_path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    temp_path = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
    with temp_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temp_path.replace(manifest_path)


def review_indices(rows: list[dict[str, str]], pending_only: bool) -> list[int]:
    if not pending_only:
        return list(range(len(rows)))
    return [idx for idx, row in enumerate(rows) if row.get("status", "").strip().lower() == STATUS_PENDING]


def _load_crop_image(crop_path: Path):
    image = cv2.imread(str(crop_path))
    if image is None:
        image = np.zeros((320, 640, 3), dtype=np.uint8)
        cv2.putText(
            image,
            "Missing crop image",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            image,
            str(crop_path),
            (20, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return image


def _draw_overlay(image, lines: list[str]):
    canvas = image.copy()
    panel_height = 28 + len(lines) * 22
    cv2.rectangle(canvas, (8, 8), (min(canvas.shape[1] - 8, 900), panel_height), (0, 0, 0), -1)
    cv2.rectangle(canvas, (8, 8), (min(canvas.shape[1] - 8, 900), panel_height), (255, 255, 255), 1)
    for idx, text in enumerate(lines):
        cv2.putText(
            canvas,
            text,
            (16, 30 + idx * 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return canvas


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Review GANomaly crop candidates with keyboard shortcuts.")
    parser.add_argument(
        "--manifest",
        default="dataset/ganomaly/review_manifest.csv",
        help="Path to review_manifest.csv.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Review all rows instead of pending rows only.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    manifest_path = Path(args.manifest).resolve()
    rows, fieldnames = load_manifest_rows(manifest_path)
    indices = review_indices(rows, pending_only=not args.all)
    if not indices:
        print("No rows to review.")
        return

    print(f"Reviewing {len(indices)} rows from {manifest_path}")
    print("Keys: [a] approve  [r] reject  [p] pending  [s] skip  [b] back  [q] save+quit")

    cursor = 0
    window_name = "GANomaly Crop Review"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    while 0 <= cursor < len(indices):
        row_index = indices[cursor]
        row = rows[row_index]
        crop_path = Path(row.get("crop_path", "")).resolve()
        image = _load_crop_image(crop_path)
        display = _draw_overlay(
            image,
            [
                f"{cursor + 1}/{len(indices)}  status={row.get('status', '')}  sample_id={row.get('sample_id', '')}",
                f"class={row.get('class_name', '')}  track={row.get('track_id', '')}  frame={row.get('frame_idx', '')}",
                f"lane={row.get('lane_id', '')}  source={row.get('source', '')}",
                "[a] approve  [r] reject  [p] pending  [s] skip  [b] back  [q] save+quit",
            ],
        )
        cv2.imshow(window_name, display)
        key = cv2.waitKey(0) & 0xFF

        if key == ord("a"):
            rows[row_index]["status"] = STATUS_APPROVED
            cursor += 1
        elif key == ord("r"):
            rows[row_index]["status"] = STATUS_REJECTED
            cursor += 1
        elif key == ord("p"):
            rows[row_index]["status"] = STATUS_PENDING
            cursor += 1
        elif key == ord("s"):
            cursor += 1
        elif key == ord("b"):
            cursor = max(0, cursor - 1)
        elif key == ord("q") or key == 27:
            break

    cv2.destroyAllWindows()
    save_manifest_rows(manifest_path, fieldnames, rows)
    print(f"Saved manifest: {manifest_path}")


if __name__ == "__main__":
    main()
