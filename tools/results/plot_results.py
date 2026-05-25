#!/usr/bin/env python3
"""Create paper plot PNGs from paper table CSVs."""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path


def _read_csv(path: Path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _bar(ax, rows, x_key, y_key, ylabel):
    labels = [row[x_key] for row in rows]
    values = [float(row[y_key] or 0.0) for row in rows]
    ax.bar(labels, values, color="#3b7c6e")
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=20)
    ax.grid(axis="y", alpha=0.25)


def _row_float(row, key):
    return float(row.get(key, 0.0) or 0.0)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", required=True, help="Experiment results root")
    args = parser.parse_args()

    os.environ.setdefault("MPLCONFIGDIR", "/tmp/entfac_matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    results_dir = Path(args.results_dir)
    paper_dir = results_dir / "paper"
    paper_dir.mkdir(parents=True, exist_ok=True)
    ablation = _read_csv(paper_dir / "table_ablation.csv")
    runtime = _read_csv(paper_dir / "table_runtime.csv")

    fig, ax = plt.subplots(figsize=(7, 4))
    _bar(ax, runtime, "Method", "Total mean ms", "Runtime (ms/frame)")
    fig.tight_layout()
    fig.savefig(paper_dir / "fig_runtime_bar.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    _bar(ax, ablation, "Method", "Retention %", "Output retention (%)")
    fig.tight_layout()
    fig.savefig(paper_dir / "fig_retention_bar.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    labels = [row["Method"] for row in ablation]
    invalid_values = [
        float(row["Would-hit invalid mask/frame"] if row["Method"] == "Naive" else row["Invalid-mask rejected/frame"])
        for row in ablation
    ]
    ax.bar(labels, invalid_values, color="#b54a3a")
    ax.set_ylabel("Invalid-region points/frame")
    ax.tick_params(axis="x", rotation=20)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(paper_dir / "fig_invalid_reduction_bar.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    edge_values = [
        float(row["Would-hit depth edge/frame"] if row["Method"] == "Naive" else row["Depth-edge rejected/frame"])
        for row in ablation
    ]
    ax.bar(labels, edge_values, color="#d8892c")
    ax.set_ylabel("Depth-edge risk points/frame")
    ax.tick_params(axis="x", rotation=20)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(paper_dir / "fig_edge_reduction_bar.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    stacks = [
        ("Invalid-mask rejected/frame", "#b54a3a"),
        ("Confidence rejected/frame", "#4f6fa8"),
        ("Depth-edge rejected/frame", "#d8892c"),
        ("Occlusion rejected/frame", "#9a4ea3"),
        ("Other rejected/frame", "#777777"),
    ]
    bottoms = [0.0] * len(ablation)
    for key, color in stacks:
        values = [_row_float(row, key) for row in ablation]
        ax.bar(labels, values, bottom=bottoms, label=key.replace("/frame", ""), color=color)
        bottoms = [a + b for a, b in zip(bottoms, values)]
    ax.set_ylabel("Rejected points/frame")
    ax.tick_params(axis="x", rotation=20)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(paper_dir / "fig_ablation_stacked_rejections.png", dpi=200)
    plt.close(fig)

    print(f"Wrote paper plots under {paper_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
