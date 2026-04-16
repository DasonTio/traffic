# Traffic Anomaly Detection

This project runs YOLO + tracking + rule-based traffic anomaly detection with GANomaly support scoring.

## Video Source Selection

The scene config now keeps both a YouTube source and a local MP4 source:

- `youtube`: configured in [configs/scene_config.yaml](configs/scene_config.yaml)
- `local`: currently set to `.video/video.mp4.mp4`

The default source mode is also configured in `configs/scene_config.yaml`:

```yaml
video:
  default: youtube
  youtube: https://www.youtube.com/watch?v=wWSSUfL2LpE
  local: .video/video.mp4.mp4
```

## Usage

Run with the default source from the config:

```powershell
python main.py --batch
```

Run with the configured YouTube source:

```powershell
python main.py --batch --source-mode youtube
```

Run with the configured local MP4:

```powershell
python main.py --batch --source-mode local
```

Override with any custom source path or URL:

```powershell
python main.py --batch --source ".video\\video.mp4.mp4"
```

`--source` has higher priority than `--source-mode`.

## Notes

- If you rename the local file, update `video.local` in `configs/scene_config.yaml`.
- For final evaluation runs, prefer the local MP4 so the inference is reproducible and does not depend on the YouTube stream.
