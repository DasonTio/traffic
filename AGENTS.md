# Repository Guidelines

## Project Structure & Module Organization
`traffic_anomaly/` contains the runtime package: `pipeline.py` orchestrates inference, `config.py` loads scene settings, and supporting modules handle rules, events, geometry, storage, GANomaly, and visualization. `main.py` is the CLI entrypoint. Keep YAML in `configs/`, utilities in `scripts/`, and tests in `tests/`. Treat `runs/`, generated `dataset/` outputs, `models/`, `checkpoints/`, and `.video/` as local artifacts, not source files.

## Build, Test, and Development Commands
Create an environment and install dependencies with `python -m venv .venv` then `source .venv/bin/activate` and `pip install -r requirements.txt`.

- `python main.py --batch --source-mode local`: run the full detection pipeline without the OpenCV preview window.
- `python main.py --batch --source "<path-or-url>"`: override the configured source for a reproducible test run.
- `pytest -q`: run the unit tests in `tests/`.
- `python scripts/train_ganomaly.py --group car`: train a GANomaly checkpoint using the paths from `configs/scene_config.yaml`.
- `python scripts/evaluate_detections.py --run <run_id>`: generate a report for a saved run under `runs/`.

## Coding Style & Naming Conventions
Follow PEP 8 with 4-space indentation and keep imports and type hints consistent with the current codebase. Use `snake_case` for functions, variables, and modules; use `PascalCase` for dataclasses and classes such as `SceneConfig` and `TrafficAnomalyPipeline`. Prefer `pathlib.Path` for filesystem logic and preserve the existing YAML-driven configuration pattern. No formatter or linter is enforced in-repo, so keep style disciplined before opening a PR.

## Testing Guidelines
Use `pytest` and name new files `tests/test_<feature>.py`. Add focused unit tests for rule changes, config parsing, and geometry helpers. There is no formal coverage gate yet, but every behavior change should ship with at least one deterministic test.

## Commit & Pull Request Guidelines
Recent history uses short imperative subjects with optional prefixes such as `feat:`, `chore:`, and scoped entries like `ganomaly:`. Follow that pattern and keep each commit limited to one concern. PRs should state the scenario changed, list config or model artifacts affected, include the exact commands run for verification, and attach screenshots only when overlay or labeling UI output changes.

## Configuration & Data Hygiene
Do not commit generated runs, local videos, checkpoints, or derived dataset sequences. When changing `configs/scene_config.yaml`, call out any path, threshold, or source-mode changes in the PR because they directly affect reproducibility.
