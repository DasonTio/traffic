from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SYSTEM_METRICS = ("precision", "recall", "f1")
APPEARANCE_METRICS = ("precision", "recall", "f1", "auroc", "auprc")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot tracker/system and GAN-vs-VAE appearance comparisons from full evaluation summaries."
    )
    parser.add_argument(
        "summaries",
        nargs="+",
        help="Path(s) to full_evaluation_summary.json. Use one for appearance-only plots or multiple for tracker/system comparison.",
    )
    parser.add_argument(
        "--labels",
        nargs="*",
        default=None,
        help="Optional labels for the summary files, in the same order.",
    )
    parser.add_argument(
        "--output-dir",
        default="reports/plots",
        help="Directory for generated PNG files.",
    )
    parser.add_argument(
        "--prefix",
        default="",
        help="Optional filename prefix such as 'ocsort' or 'bytetrack'.",
    )
    return parser.parse_args()


def load_summary(path: str | Path) -> dict:
    summary_path = Path(path).resolve()
    with summary_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def infer_label(summary_path: Path, summary: dict) -> str:
    system = summary.get("system") or {}
    run_dir = system.get("run_dir", "")
    if run_dir:
        return Path(run_dir).name
    return summary_path.parent.name


def _prefixed_name(prefix: str, stem: str) -> str:
    normalized = prefix.strip().replace(" ", "_")
    return f"{normalized}_{stem}" if normalized else stem


def plot_system_metrics(summaries: list[dict], labels: list[str], output_dir: Path, prefix: str = "") -> Path | None:
    rows: list[list[float]] = []
    filtered_labels: list[str] = []
    for summary, label in zip(summaries, labels):
        system = summary.get("system")
        if not system:
            continue
        rows.append([float(system.get(metric, 0.0)) for metric in SYSTEM_METRICS])
        filtered_labels.append(label)

    if not rows:
        return None

    values = np.asarray(rows, dtype=float)
    x = np.arange(len(filtered_labels))
    width = 0.22

    fig, ax = plt.subplots(figsize=(10, 5))
    for idx, metric in enumerate(SYSTEM_METRICS):
        ax.bar(x + (idx - 1) * width, values[:, idx], width=width, label=metric.upper())

    ax.set_title("System Metrics by Run")
    ax.set_ylabel("Score")
    ax.set_ylim(0.0, 1.05)
    ax.set_xticks(x)
    ax.set_xticklabels(filtered_labels, rotation=15, ha="right")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    fig.tight_layout()

    output_path = output_dir / f"{_prefixed_name(prefix, 'system_metrics')}.png"
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def plot_appearance_metrics(summary: dict, output_dir: Path, prefix: str = "") -> Path | None:
    if "ganomaly" not in summary or "vae" not in summary:
        return None

    model_names = ["ganomaly", "vae"]
    values = np.asarray(
        [[float(summary[model].get(metric, 0.0)) for metric in APPEARANCE_METRICS] for model in model_names],
        dtype=float,
    )
    x = np.arange(len(APPEARANCE_METRICS))
    width = 0.35

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(x - width / 2, values[0], width=width, label="GANomaly")
    ax.bar(x + width / 2, values[1], width=width, label="VAE")

    title_prefix = f"{prefix.strip()} " if prefix.strip() else ""
    ax.set_title(f"{title_prefix}Appearance Model Comparison")
    ax.set_ylabel("Score")
    ax.set_ylim(0.0, 1.05)
    ax.set_xticks(x)
    ax.set_xticklabels([metric.upper() for metric in APPEARANCE_METRICS])
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    fig.tight_layout()

    output_path = output_dir / f"{_prefixed_name(prefix, 'appearance_metrics')}.png"
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def plot_appearance_per_group(summary: dict, output_dir: Path, prefix: str = "") -> Path | None:
    if "ganomaly" not in summary or "vae" not in summary:
        return None

    groups = sorted(set(summary["ganomaly"].get("per_group", {})) | set(summary["vae"].get("per_group", {})))
    if not groups:
        return None

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    metrics = ("f1", "auroc")
    model_titles = ("GANomaly", "VAE")

    for ax, model_name, title in zip(axes, ("ganomaly", "vae"), model_titles):
        model_groups = summary[model_name].get("per_group", {})
        group_values = np.asarray(
            [[float(model_groups.get(group, {}).get(metric, 0.0)) for metric in metrics] for group in groups],
            dtype=float,
        )
        x = np.arange(len(groups))
        width = 0.35
        for idx, metric in enumerate(metrics):
            ax.bar(x + (idx - 0.5) * width, group_values[:, idx], width=width, label=metric.upper())
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(groups)
        ax.set_ylim(0.0, 1.05)
        ax.grid(axis="y", linestyle="--", alpha=0.35)

    axes[0].set_ylabel("Score")
    axes[1].legend()
    title_prefix = f"{prefix.strip()} " if prefix.strip() else ""
    fig.suptitle(f"{title_prefix}Appearance Metrics by Class Group")
    fig.tight_layout()

    output_path = output_dir / f"{_prefixed_name(prefix, 'appearance_per_group')}.png"
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def main() -> None:
    args = parse_args()
    summary_paths = [Path(path).resolve() for path in args.summaries]
    summaries = [load_summary(path) for path in summary_paths]

    if args.labels and len(args.labels) != len(summary_paths):
        raise SystemExit("--labels must have the same number of values as summaries.")

    labels = args.labels or [infer_label(path, summary) for path, summary in zip(summary_paths, summaries)]
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    outputs = [
        plot_system_metrics(summaries, labels, output_dir, prefix=args.prefix),
        plot_appearance_metrics(summaries[0], output_dir, prefix=args.prefix),
        plot_appearance_per_group(summaries[0], output_dir, prefix=args.prefix),
    ]

    for output in outputs:
        if output is not None:
            print(output)


if __name__ == "__main__":
    main()
