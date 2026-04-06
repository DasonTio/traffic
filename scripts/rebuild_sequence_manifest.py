import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from traffic_anomaly.storage import sync_sequence_manifests


def main() -> None:
    dataset_dir = Path("dataset").resolve()
    sync_sequence_manifests(dataset_dir)
    print(f"Rebuilt manifests under {dataset_dir}")


if __name__ == "__main__":
    main()
