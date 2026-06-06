#!/usr/bin/env python3
"""Experiment metrics helpers for Sensor Fusion ablation runs."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence


DROP_REASONS = {
    "none",
    "pair_dt_too_large",
    "queue_full",
    "missing_tf",
    "missing_camera_info",
    "empty_cloud",
    "no_points_in_front",
    "no_points_projected",
    "exception",
}

SUMMARY_FIELDS = [
    "bag_name",
    "variant_name",
    "num_frames_seen",
    "num_frames_accepted",
    "pair_drop_rate",
    "mean_pair_dt_sec",
    "std_pair_dt_sec",
    "p95_pair_dt_sec",
    "mean_input_points",
    "mean_projected_points",
    "mean_output_points",
    "mean_invalid_mask_rejected",
    "mean_confidence_rejected",
    "mean_depth_edge_rejected",
    "mean_occlusion_rejected",
    "mean_other_rejected",
    "mean_output_retention_ratio",
    "mean_runtime_total_ms",
    "p95_runtime_total_ms",
    "mean_runtime_projection_ms",
    "mean_runtime_mask_ms",
    "mean_runtime_rasterize_ms",
    "mean_runtime_depth_edge_ms",
    "mean_runtime_occlusion_ms",
    "mean_runtime_publish_ms",
    "effective_fps_mean",
    "effective_fps_p95",
    "mean_would_hit_invalid_mask",
    "mean_would_hit_depth_edge",
    "mean_would_fail_occlusion",
    "mean_projection_projected_ratio",
    "mean_projection_in_front_ratio",
    "mean_projection_in_image_ratio",
    "mean_projection_invalid_mask_hit_ratio",
    "mean_projection_confidence_rejection_ratio",
    "mean_projection_depth_edge_rejection_ratio",
    "mean_projection_occlusion_rejection_ratio",
    "mean_projection_health_score",
    "mean_projection_suppressed",
    "mean_projection_rejected",
    "mean_projection_accepted",
]


@dataclass
class FrameMetrics:
    """One row of per-frame ablation metrics."""

    bag_name: str = ""
    variant_name: str = ""
    frame_index: int = 0
    stamp_semantic: float = 0.0
    stamp_cloud: float = 0.0
    pair_dt_sec: float = 0.0
    pair_accepted: int = 1
    drop_reason: str = "none"
    num_input_points: int = 0
    num_points_in_front: int = 0
    num_points_projected_in_image: int = 0
    num_rejected_invalid_mask: int = 0
    num_rejected_confidence: int = 0
    num_rejected_depth_edge: int = 0
    num_rejected_occlusion: int = 0
    num_rejected_other: int = 0
    num_output_points: int = 0
    output_retention_ratio: float = 0.0
    runtime_total_ms: float = 0.0
    runtime_projection_ms: float = 0.0
    runtime_mask_ms: float = 0.0
    runtime_rasterize_ms: float = 0.0
    runtime_depth_edge_ms: float = 0.0
    runtime_occlusion_ms: float = 0.0
    runtime_publish_ms: float = 0.0
    num_would_hit_invalid_mask: int = 0
    would_hit_invalid_mask_ratio: float = 0.0
    num_would_hit_depth_edge: int = 0
    would_hit_depth_edge_ratio: float = 0.0
    num_would_fail_occlusion: int = 0
    would_fail_occlusion_ratio: float = 0.0
    projection_projected_ratio: float = 0.0
    projection_in_front_ratio: float = 0.0
    projection_in_image_ratio: float = 0.0
    projection_invalid_mask_hit_ratio: float = 0.0
    projection_confidence_rejection_ratio: float = 0.0
    projection_depth_edge_rejection_ratio: float = 0.0
    projection_occlusion_rejection_ratio: float = 0.0
    projection_health_score: float = 0.0
    num_projection_suppressed: int = 0
    num_projection_rejected: int = 0
    num_projection_accepted: int = 0

    @classmethod
    def fieldnames(cls) -> List[str]:
        return [field.name for field in fields(cls)]

    def to_row(self) -> Dict[str, object]:
        if self.drop_reason not in DROP_REASONS:
            raise ValueError(f"unsupported drop_reason: {self.drop_reason}")
        row = {name: getattr(self, name) for name in self.fieldnames()}
        projected = max(int(self.num_points_projected_in_image), 1)
        row["would_hit_invalid_mask_ratio"] = float(self.num_would_hit_invalid_mask) / projected
        row["would_hit_depth_edge_ratio"] = float(self.num_would_hit_depth_edge) / projected
        row["would_fail_occlusion_ratio"] = float(self.num_would_fail_occlusion) / projected
        return row


class MetricsCsvLogger:
    """Append per-frame metrics rows to a CSV file."""

    def __init__(self, path: Path | str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        exists = self.path.exists() and self.path.stat().st_size > 0
        self._handle = self.path.open("a", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._handle, fieldnames=FrameMetrics.fieldnames())
        if not exists:
            self._writer.writeheader()
            self._handle.flush()

    def write(self, metrics: FrameMetrics) -> None:
        self._writer.writerow(metrics.to_row())
        self._handle.flush()

    def close(self) -> None:
        self._handle.close()


def read_metrics_csv(path: Path | str) -> List[Dict[str, object]]:
    path = Path(path)
    with path.open(newline="", encoding="utf-8") as handle:
        return [_coerce_metrics_row(row) for row in csv.DictReader(handle)]


def summarize_metrics_file(path: Path | str) -> Dict[str, object]:
    path = Path(path)
    rows = read_metrics_csv(path)
    summary = summarize_metrics_rows(rows)
    write_summary_files(path.parent, summary)
    return summary


def summarize_metrics_rows(rows: Sequence[Mapping[str, object]]) -> Dict[str, object]:
    if not rows:
        return _empty_summary()

    first = rows[0]
    accepted = [row for row in rows if int(row.get("pair_accepted", 0)) == 1]
    num_frames_seen = len(rows)
    num_frames_accepted = len(accepted)
    pair_dt_values = [_float(row, "pair_dt_sec") for row in rows]
    runtime_values = [_float(row, "runtime_total_ms") for row in rows]
    mean_runtime_total_ms = _mean(runtime_values)
    p95_runtime_total_ms = _percentile(runtime_values, 95.0)

    summary = {
        "bag_name": str(first.get("bag_name", "")),
        "variant_name": str(first.get("variant_name", "")),
        "num_frames_seen": num_frames_seen,
        "num_frames_accepted": num_frames_accepted,
        "pair_drop_rate": 1.0 - num_frames_accepted / max(num_frames_seen, 1),
        "mean_pair_dt_sec": _mean(pair_dt_values),
        "std_pair_dt_sec": _std(pair_dt_values),
        "p95_pair_dt_sec": _percentile(pair_dt_values, 95.0),
        "mean_input_points": _mean_field(accepted, "num_input_points"),
        "mean_projected_points": _mean_field(accepted, "num_points_projected_in_image"),
        "mean_output_points": _mean_field(accepted, "num_output_points"),
        "mean_invalid_mask_rejected": _mean_field(accepted, "num_rejected_invalid_mask"),
        "mean_confidence_rejected": _mean_field(accepted, "num_rejected_confidence"),
        "mean_depth_edge_rejected": _mean_field(accepted, "num_rejected_depth_edge"),
        "mean_occlusion_rejected": _mean_field(accepted, "num_rejected_occlusion"),
        "mean_other_rejected": _mean_field(accepted, "num_rejected_other"),
        "mean_output_retention_ratio": _mean_field(accepted, "output_retention_ratio"),
        "mean_runtime_total_ms": mean_runtime_total_ms,
        "p95_runtime_total_ms": p95_runtime_total_ms,
        "mean_runtime_projection_ms": _mean_field(accepted, "runtime_projection_ms"),
        "mean_runtime_mask_ms": _mean_field(accepted, "runtime_mask_ms"),
        "mean_runtime_depth_edge_ms": _mean_field(accepted, "runtime_depth_edge_ms"),
        "mean_runtime_occlusion_ms": _mean_field(accepted, "runtime_occlusion_ms"),
        "mean_runtime_publish_ms": _mean_field(accepted, "runtime_publish_ms"),
        "effective_fps_mean": _fps(mean_runtime_total_ms),
        "effective_fps_p95": _fps(p95_runtime_total_ms),
        "mean_would_hit_invalid_mask": _mean_field(accepted, "num_would_hit_invalid_mask"),
        "mean_would_hit_depth_edge": _mean_field(accepted, "num_would_hit_depth_edge"),
        "mean_would_fail_occlusion": _mean_field(accepted, "num_would_fail_occlusion"),
        "mean_projection_projected_ratio": _mean_field(accepted, "projection_projected_ratio"),
        "mean_projection_in_front_ratio": _mean_field(accepted, "projection_in_front_ratio"),
        "mean_projection_in_image_ratio": _mean_field(accepted, "projection_in_image_ratio"),
        "mean_projection_invalid_mask_hit_ratio": _mean_field(accepted, "projection_invalid_mask_hit_ratio"),
        "mean_projection_confidence_rejection_ratio": _mean_field(accepted, "projection_confidence_rejection_ratio"),
        "mean_projection_depth_edge_rejection_ratio": _mean_field(accepted, "projection_depth_edge_rejection_ratio"),
        "mean_projection_occlusion_rejection_ratio": _mean_field(accepted, "projection_occlusion_rejection_ratio"),
        "mean_projection_health_score": _mean_field(accepted, "projection_health_score"),
        "mean_projection_suppressed": _mean_field(accepted, "num_projection_suppressed"),
        "mean_projection_rejected": _mean_field(accepted, "num_projection_rejected"),
        "mean_projection_accepted": _mean_field(accepted, "num_projection_accepted"),
    }
    return _round_summary(summary)


def write_summary_files(out_dir: Path | str, summary: Mapping[str, object]) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(dict(summary), handle, indent=2, sort_keys=True)
        handle.write("\n")
    with (out_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerow({name: summary.get(name, 0.0) for name in SUMMARY_FIELDS})


def summarize_results_tree(results_dir: Path | str) -> List[Dict[str, object]]:
    results_dir = Path(results_dir)
    summaries: List[Dict[str, object]] = []
    for metrics_path in sorted(results_dir.glob("*/*/metrics_per_frame.csv")):
        summaries.append(summarize_metrics_file(metrics_path))

    all_path = results_dir / "all_results_summary.csv"
    all_path.parent.mkdir(parents=True, exist_ok=True)
    with all_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        for summary in summaries:
            writer.writerow({name: summary.get(name, 0.0) for name in SUMMARY_FIELDS})
    return summaries


def write_results_tables(results_dir: Path | str) -> None:
    results_dir = Path(results_dir)
    summaries = _load_or_build_summaries(results_dir)
    paper_dir = results_dir / "paper"
    paper_dir.mkdir(parents=True, exist_ok=True)
    _write_ablation_table(paper_dir / "table_ablation.csv", summaries)
    _write_runtime_table(paper_dir / "table_runtime.csv", summaries)
    _write_bags_table(paper_dir / "table_bags.csv", summaries)


def _write_ablation_table(path: Path, summaries: Sequence[Mapping[str, object]]) -> None:
    fields_out = [
        "Method",
        "Projected pts/frame",
        "Would-hit invalid mask/frame",
        "Invalid-mask rejected/frame",
        "Confidence rejected/frame",
        "Would-hit depth edge/frame",
        "Depth-edge rejected/frame",
        "Would-fail occlusion/frame",
        "Occlusion rejected/frame",
        "Other rejected/frame",
        "Output pts/frame",
        "Retention %",
        "Runtime ms/frame",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields_out)
        writer.writeheader()
        for summary in _ordered_method_summaries(summaries):
            writer.writerow(
                {
                    "Method": _method_label(str(summary["variant_name"])),
                    "Projected pts/frame": _fmt(summary["mean_projected_points"]),
                    "Would-hit invalid mask/frame": _fmt(summary.get("mean_would_hit_invalid_mask", 0.0)),
                    "Invalid-mask rejected/frame": _fmt(summary["mean_invalid_mask_rejected"]),
                    "Confidence rejected/frame": _fmt(summary.get("mean_confidence_rejected", 0.0)),
                    "Would-hit depth edge/frame": _fmt(summary.get("mean_would_hit_depth_edge", 0.0)),
                    "Depth-edge rejected/frame": _fmt(summary["mean_depth_edge_rejected"]),
                    "Would-fail occlusion/frame": _fmt(summary.get("mean_would_fail_occlusion", 0.0)),
                    "Occlusion rejected/frame": _fmt(summary["mean_occlusion_rejected"]),
                    "Other rejected/frame": _fmt(summary.get("mean_other_rejected", 0.0)),
                    "Output pts/frame": _fmt(summary["mean_output_points"]),
                    "Retention %": _fmt(100.0 * float(summary["mean_output_retention_ratio"])),
                    "Runtime ms/frame": _fmt(summary["mean_runtime_total_ms"]),
                }
            )


def _write_runtime_table(path: Path, summaries: Sequence[Mapping[str, object]]) -> None:
    fields_out = [
        "Method",
        "Projection ms",
        "Mask ms",
        "Depth-edge ms",
        "Occlusion ms",
        "Publish ms",
        "Total mean ms",
        "Total p95 ms",
        "FPS mean",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields_out)
        writer.writeheader()
        for summary in _ordered_method_summaries(summaries):
            writer.writerow(
                {
                    "Method": _method_label(str(summary["variant_name"])),
                    "Projection ms": _fmt(summary["mean_runtime_projection_ms"]),
                    "Mask ms": _fmt(summary["mean_runtime_mask_ms"]),
                    "Depth-edge ms": _fmt(summary["mean_runtime_depth_edge_ms"]),
                    "Occlusion ms": _fmt(summary["mean_runtime_occlusion_ms"]),
                    "Publish ms": _fmt(summary.get("mean_runtime_publish_ms", 0.0)),
                    "Total mean ms": _fmt(summary["mean_runtime_total_ms"]),
                    "Total p95 ms": _fmt(summary["p95_runtime_total_ms"]),
                    "FPS mean": _fmt(summary["effective_fps_mean"]),
                }
            )


def _write_bags_table(path: Path, summaries: Sequence[Mapping[str, object]]) -> None:
    fields_out = [
        "Bag",
        "Frames",
        "Accepted frames",
        "Pair drop %",
        "Mean pair dt",
        "P95 pair dt",
        "Mean projected pts",
        "Mean output pts",
        "Mean runtime ms",
    ]
    grouped: Dict[str, List[Mapping[str, object]]] = {}
    for summary in summaries:
        grouped.setdefault(str(summary["bag_name"]), []).append(summary)

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields_out)
        writer.writeheader()
        for bag_name in sorted(grouped):
            bag_summaries = grouped[bag_name]
            writer.writerow(
                {
                    "Bag": bag_name,
                    "Frames": _fmt(_mean_field(bag_summaries, "num_frames_seen"), 0),
                    "Accepted frames": _fmt(_mean_field(bag_summaries, "num_frames_accepted"), 0),
                    "Pair drop %": _fmt(100.0 * _mean_field(bag_summaries, "pair_drop_rate")),
                    "Mean pair dt": _fmt(_mean_field(bag_summaries, "mean_pair_dt_sec"), 6),
                    "P95 pair dt": _fmt(_mean_field(bag_summaries, "p95_pair_dt_sec"), 6),
                    "Mean projected pts": _fmt(_mean_field(bag_summaries, "mean_projected_points")),
                    "Mean output pts": _fmt(_mean_field(bag_summaries, "mean_output_points")),
                    "Mean runtime ms": _fmt(_mean_field(bag_summaries, "mean_runtime_total_ms")),
                }
            )


def _load_or_build_summaries(results_dir: Path) -> List[Dict[str, object]]:
    summary_path = results_dir / "all_results_summary.csv"
    if summary_path.exists():
        with summary_path.open(newline="", encoding="utf-8") as handle:
            return [_coerce_summary_row(row) for row in csv.DictReader(handle)]
    return summarize_results_tree(results_dir)


def _ordered_method_summaries(summaries: Sequence[Mapping[str, object]]) -> List[Dict[str, object]]:
    grouped: Dict[str, List[Mapping[str, object]]] = {}
    for summary in summaries:
        grouped.setdefault(str(summary["variant_name"]), []).append(summary)

    out: List[Dict[str, object]] = []
    for variant in sorted(grouped, key=_variant_order):
        rows = grouped[variant]
        merged = dict(rows[0])
        for name in SUMMARY_FIELDS:
            if name in {"bag_name", "variant_name"}:
                continue
            merged[name] = _mean_field(rows, name)
        merged["bag_name"] = "all"
        merged["variant_name"] = variant
        out.append(merged)
    return out


def _empty_summary() -> Dict[str, object]:
    summary = {name: 0.0 for name in SUMMARY_FIELDS}
    summary["bag_name"] = ""
    summary["variant_name"] = ""
    summary["num_frames_seen"] = 0
    summary["num_frames_accepted"] = 0
    return summary


def _round_summary(summary: Mapping[str, object]) -> Dict[str, object]:
    out: Dict[str, object] = {}
    for key, value in summary.items():
        if isinstance(value, float):
            out[key] = round(value, 12)
        else:
            out[key] = value
    return out


def _coerce_metrics_row(row: Mapping[str, str]) -> Dict[str, object]:
    out: Dict[str, object] = {}
    for name in FrameMetrics.fieldnames():
        value = row.get(name, "")
        if name in {"bag_name", "variant_name", "drop_reason"}:
            out[name] = value
        elif name in {
            "frame_index",
            "pair_accepted",
            "num_input_points",
            "num_points_in_front",
            "num_points_projected_in_image",
            "num_rejected_invalid_mask",
            "num_rejected_confidence",
            "num_rejected_depth_edge",
            "num_rejected_occlusion",
            "num_rejected_other",
            "num_output_points",
            "num_would_hit_invalid_mask",
            "num_would_hit_depth_edge",
            "num_would_fail_occlusion",
        }:
            out[name] = int(float(value or 0))
        else:
            out[name] = float(value or 0.0)
    return out


def _coerce_summary_row(row: Mapping[str, str]) -> Dict[str, object]:
    out: Dict[str, object] = {}
    for name in SUMMARY_FIELDS:
        value = row.get(name, "")
        if name in {"bag_name", "variant_name"}:
            out[name] = value
        elif name in {"num_frames_seen", "num_frames_accepted"}:
            out[name] = int(float(value or 0))
        else:
            out[name] = float(value or 0.0)
    return out

def _variant_order(variant: str) -> tuple[int, str]:
    order = {
        "naive": 0,
        "mask": 1,
        "edge": 2,
        "occlusion": 3,
        "mask_edge": 4,
        "mask_occlusion": 5,
        "edge_occlusion": 6,
        "full": 7,
        "patch1_full": 8,
        "patch3_full": 9,
        "patch5_full": 10,
        "confidence_sweep": 11,
    }
    return order.get(variant, 100), variant


def _method_label(variant: str) -> str:
    labels = {
        "naive": "Naive",
        "mask": "Mask",
        "edge": "Edge",
        "occlusion": "Occlusion",
        "mask_edge": "Mask + edge",
        "mask_occlusion": "Mask + occlusion",
        "edge_occlusion": "Edge + occlusion",
        "full": "Full",
        "patch1_full": "Patch 1 full",
        "patch3_full": "Patch 3 full",
        "patch5_full": "Patch 5 full",
        "confidence_sweep": "Confidence sweep",
    }
    return labels.get(variant, variant.replace("_", " ").title())


def _float(row: Mapping[str, object], key: str) -> float:
    return float(row.get(key, 0.0) or 0.0)


def _mean_field(rows: Sequence[Mapping[str, object]], key: str) -> float:
    return _mean([_float(row, key) for row in rows])


def _mean(values: Iterable[float]) -> float:
    vals = [float(v) for v in values]
    if not vals:
        return 0.0
    return sum(vals) / len(vals)


def _std(values: Iterable[float]) -> float:
    vals = [float(v) for v in values]
    if len(vals) < 2:
        return 0.0
    mean = _mean(vals)
    return (sum((v - mean) ** 2 for v in vals) / (len(vals) - 1)) ** 0.5


def _percentile(values: Iterable[float], percentile: float) -> float:
    vals = sorted(float(v) for v in values)
    if not vals:
        return 0.0
    if len(vals) == 1:
        return vals[0]
    rank = (len(vals) - 1) * float(percentile) / 100.0
    lo = int(rank)
    hi = min(lo + 1, len(vals) - 1)
    frac = rank - lo
    return vals[lo] * (1.0 - frac) + vals[hi] * frac


def _fps(runtime_ms: float) -> float:
    runtime_ms = float(runtime_ms)
    if runtime_ms <= 0.0:
        return 0.0
    return 1000.0 / runtime_ms


def _fmt(value: object, digits: int = 3) -> str:
    return f"{float(value):.{digits}f}"
