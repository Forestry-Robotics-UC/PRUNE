#!/usr/bin/env python3
"""Tests for the ICNF results-dir validation report."""

from __future__ import annotations

import csv
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ROS_SRC = ROOT / "prune_ros"
if str(ROS_SRC) not in sys.path:
    sys.path.insert(0, str(ROS_SRC))
import prune_ros as _prune_ros_pkg
_inner_pkg = str(ROS_SRC / "prune_ros")
if hasattr(_prune_ros_pkg, "__path__") and _inner_pkg not in list(_prune_ros_pkg.__path__):
    _prune_ros_pkg.__path__.append(_inner_pkg)

from prune_ros.diagnostics.experiment_metrics import FrameMetrics  # noqa: E402
from tools.validation.icnf_report import build_icnf_report  # noqa: E402


def _write_metrics_csv(path: Path, rows: list[FrameMetrics]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FrameMetrics.fieldnames())
        writer.writeheader()
        for row in rows:
            writer.writerow(row.to_row())


def _write_summary_csv(path: Path, *, bag_name: str, variant_name: str) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["bag_name", "variant_name"])
        writer.writeheader()
        writer.writerow({"bag_name": bag_name, "variant_name": variant_name})


def _make_row(
    *,
    bag_name: str,
    variant_name: str,
    pair_dt_sec: float,
    projected: int,
    output: int,
    invalid_would: int,
    invalid_rejected: int,
    confidence_rejected: int,
    depth_edge_would: int,
    depth_edge_rejected: int,
    occlusion_would: int,
    occlusion_rejected: int,
) -> FrameMetrics:
    return FrameMetrics(
        bag_name=bag_name,
        variant_name=variant_name,
        frame_index=0,
        stamp_semantic=1.0,
        stamp_cloud=1.0 - pair_dt_sec,
        pair_dt_sec=pair_dt_sec,
        pair_accepted=1,
        drop_reason="none",
        num_input_points=projected,
        num_points_in_front=projected,
        num_points_projected_in_image=projected,
        num_rejected_invalid_mask=invalid_rejected,
        num_rejected_confidence=confidence_rejected,
        num_rejected_depth_edge=depth_edge_rejected,
        num_rejected_occlusion=occlusion_rejected,
        num_rejected_other=0,
        num_output_points=output,
        output_retention_ratio=float(output) / max(projected, 1),
        runtime_total_ms=12.5,
        runtime_projection_ms=2.0,
        runtime_mask_ms=0.5,
        runtime_rasterize_ms=3.0,
        runtime_depth_edge_ms=1.0,
        runtime_occlusion_ms=1.2,
        runtime_publish_ms=0.9,
        num_would_hit_invalid_mask=invalid_would,
        would_hit_invalid_mask_ratio=float(invalid_would) / max(projected, 1),
        num_would_hit_depth_edge=depth_edge_would,
        would_hit_depth_edge_ratio=float(depth_edge_would) / max(projected, 1),
        num_would_fail_occlusion=occlusion_would,
        would_fail_occlusion_ratio=float(occlusion_would) / max(projected, 1),
    )


class IcnfValidationTests(unittest.TestCase):
    def test_full_variant_report_passes_when_enabled_gates_are_observed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "bag_a" / "full"
            rows = [
                _make_row(
                    bag_name="bag_a",
                    variant_name="full",
                    pair_dt_sec=0.01,
                    projected=12,
                    output=4,
                    invalid_would=3,
                    invalid_rejected=3,
                    confidence_rejected=2,
                    depth_edge_would=2,
                    depth_edge_rejected=2,
                    occlusion_would=1,
                    occlusion_rejected=1,
                )
            ]
            _write_metrics_csv(run_dir / "metrics_per_frame.csv", rows)
            _write_summary_csv(run_dir / "summary.csv", bag_name="bag_a", variant_name="full")

            report = build_icnf_report(run_dir)

            self.assertTrue(report["overall_pass"])
            self.assertEqual(report["runs"][0]["variant_name"], "full")
            self.assertEqual(report["checks"]["sync_pair"]["status"], "pass")
            self.assertEqual(report["checks"]["projection"]["status"], "pass")
            self.assertEqual(report["checks"]["invalid_mask"]["status"], "pass")
            self.assertEqual(report["checks"]["depth_edge"]["status"], "pass")
            self.assertEqual(report["checks"]["occlusion"]["status"], "pass")
            self.assertEqual(report["checks"]["confidence"]["status"], "pass")
            self.assertEqual(report["checks"]["suppression_vs_filtering"]["status"], "pass")

    def test_naive_variant_report_marks_enabled_gates_as_suppressed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "bag_a" / "naive"
            rows = [
                _make_row(
                    bag_name="bag_a",
                    variant_name="naive",
                    pair_dt_sec=0.01,
                    projected=10,
                    output=10,
                    invalid_would=2,
                    invalid_rejected=0,
                    confidence_rejected=1,
                    depth_edge_would=3,
                    depth_edge_rejected=0,
                    occlusion_would=1,
                    occlusion_rejected=0,
                )
            ]
            _write_metrics_csv(run_dir / "metrics_per_frame.csv", rows)
            _write_summary_csv(run_dir / "summary.csv", bag_name="bag_a", variant_name="naive")

            report = build_icnf_report(run_dir)

            self.assertTrue(report["overall_pass"])
            self.assertEqual(report["checks"]["invalid_mask"]["mode"], "suppressed")
            self.assertEqual(report["checks"]["depth_edge"]["mode"], "suppressed")
            self.assertEqual(report["checks"]["occlusion"]["mode"], "suppressed")


if __name__ == "__main__":
    unittest.main()
