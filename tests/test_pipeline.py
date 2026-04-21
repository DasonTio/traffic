from traffic_anomaly.pipeline import TrafficAnomalyPipeline


class FakeCapture:
    def __init__(self, opened: bool = True):
        self._opened = opened

    def isOpened(self) -> bool:
        return self._opened


def test_pipeline_falls_back_to_local_source(monkeypatch, tmp_path):
    local_video = tmp_path / "video.mp4"
    local_video.write_bytes(b"not-a-real-video")

    pipeline = TrafficAnomalyPipeline("configs/scene_config.yaml", display=False)
    pipeline.scene.video_source = "https://www.youtube.com/watch?v=test"
    pipeline.scene.video_source_mode = "youtube"
    pipeline.scene.video_sources["local"] = str(local_video)

    def fake_open_capture(source: str, resolution: str | None):
        if source.startswith("http"):
            raise RuntimeError("youtube unavailable")
        return FakeCapture(opened=True)

    monkeypatch.setattr("traffic_anomaly.pipeline.open_capture", fake_open_capture)

    cap = pipeline._open_video_capture()

    assert cap.isOpened()
    assert pipeline.scene.video_source == str(local_video)
    assert pipeline.scene.video_source_mode == "local-fallback"
