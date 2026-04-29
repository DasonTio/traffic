from pathlib import Path

import yaml

from scripts.compare_video_sources import write_scaled_scene_config


def test_write_scaled_scene_config_scales_geometry_and_outputs(tmp_path):
    base_config = tmp_path / "scene.yaml"
    source = tmp_path / "source.mp4"
    source.write_bytes(b"video")
    base_config.write_text(
        """
camera_id: cam
video:
  default: local
  local: old.mp4
model:
  weights: yolo11n.pt
tracking:
  tracker_config: bytetrack.yaml
output:
  run_root: runs
  dataset_dir: dataset
homography:
  src_points:
    - [10, 20]
    - [30, 40]
  dst_points:
    - [1, 2]
    - [3, 4]
lanes:
  - id: lane
    category: fast
    direction: [1, 0]
    polygon:
      - [5, 6]
      - [7, 8]
""",
        encoding="utf-8",
    )

    output = write_scaled_scene_config(
        base_config=base_config,
        output_path=tmp_path / "scaled.yaml",
        source_path=source,
        scale_x=2.0,
        scale_y=3.0,
        run_root=tmp_path / "runs",
        dataset_dir=tmp_path / "dataset",
    )

    scaled = yaml.safe_load(Path(output).read_text(encoding="utf-8"))

    assert scaled["video"]["local"] == str(source.resolve())
    assert scaled["output"]["run_root"] == str((tmp_path / "runs").resolve())
    assert scaled["output"]["dataset_dir"] == str((tmp_path / "dataset").resolve())
    assert scaled["homography"]["src_points"] == [[20.0, 60.0], [60.0, 120.0]]
    assert scaled["homography"]["dst_points"] == [[1, 2], [3, 4]]
    assert scaled["lanes"][0]["polygon"] == [[10, 18], [14, 24]]
