#!/usr/bin/env python3
"""Tests for ablation results metrics helpers."""

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROS_SRC = Path(__file__).resolve().parents[2] / "prune_ros"
if str(ROS_SRC) not in sys.path:
    sys.path.insert(0, str(ROS_SRC))

from prune_ros.diagnostics import (  # noqa: E402
    FrameMetrics,
    MetricsCsvLogger,
    summarize_metrics_file,
    write_results_tables,
)


def _write_metrics(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FrameMetrics.fieldnames())
        writer.writeheader()
        for row in rows:
            writer.writerow(FrameMetrics(**row).to_row())


class ExperimentMetricsTests(unittest.TestCase):
    def test_frame_metrics_accepts_queue_full_drop_reason(self):
        metrics = FrameMetrics(
            bag_name="bag_a",
            variant_name="full",
            pair_accepted=0,
            drop_reason="queue_full",
        )

        row = metrics.to_row()

        self.assertEqual(row["drop_reason"], "queue_full")

    def test_frame_metrics_computes_ratio_fields_on_serialization(self):
        metrics = FrameMetrics(
            bag_name="bag_a",
            variant_name="full",
            num_points_projected_in_image=100,
            num_would_hit_invalid_mask=25,
            num_would_hit_depth_edge=10,
            num_would_fail_occlusion=5,
        )

        row = metrics.to_row()

        self.assertEqual(row["would_hit_invalid_mask_ratio"], 0.25)
        self.assertEqual(row["would_hit_depth_edge_ratio"], 0.1)
        self.assertEqual(row["would_fail_occlusion_ratio"], 0.05)

    def test_metrics_csv_logger_writes_required_schema(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_path = Path(tmp) / "bag_a" / "full" / "metrics_per_frame.csv"
            logger = MetricsCsvLogger(out_path)

            logger.write(
                FrameMetrics(
                    bag_name="bag_a",
                    variant_name="full",
                    frame_index=7,
                    stamp_semantic=10.0,
                    stamp_cloud=10.02,
                    pair_dt_sec=0.02,
                    pair_accepted=1,
                    drop_reason="none",
                    num_input_points=100,
                    num_points_in_front=80,
                    num_points_projected_in_image=50,
                    num_rejected_invalid_mask=5,
                    num_rejected_confidence=2,
                    num_rejected_depth_edge=3,
                    num_rejected_occlusion=4,
                    num_rejected_other=1,
                    num_output_points=35,
                    output_retention_ratio=0.7,
                    runtime_total_ms=12.0,
                    runtime_projection_ms=4.0,
                    runtime_mask_ms=1.0,
                    runtime_depth_edge_ms=2.0,
                    runtime_occlusion_ms=3.0,
                    runtime_publish_ms=2.0,
                    num_would_hit_invalid_mask=5,
                    would_hit_invalid_mask_ratio=0.1,
                    num_would_hit_depth_edge=3,
                    would_hit_depth_edge_ratio=0.06,
                    num_would_fail_occlusion=4,
                    would_fail_occlusion_ratio=0.08,
                    projection_projected_ratio=0.5,
                    projection_in_front_ratio=0.8,
                    projection_in_image_ratio=0.625,
                    projection_invalid_mask_hit_ratio=0.1,
                    projection_confidence_rejection_ratio=0.04,
                    projection_depth_edge_rejection_ratio=0.06,
                    projection_occlusion_rejection_ratio=0.08,
                    projection_health_score=0.86,
                    num_projection_suppressed=15,
                    num_projection_rejected=15,
                    num_projection_accepted=35,
                )
            )
            logger.close()

            with out_path.open(newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                rows = list(reader)

            self.assertEqual(reader.fieldnames, FrameMetrics.fieldnames())
            self.assertEqual(rows[0]["bag_name"], "bag_a")
            self.assertEqual(rows[0]["variant_name"], "full")
            self.assertEqual(rows[0]["num_would_hit_invalid_mask"], "5")
            self.assertEqual(rows[0]["projection_health_score"], "0.86")
            self.assertEqual(rows[0]["num_projection_accepted"], "35")

    def test_summarize_metrics_file_writes_summary_csv_and_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            metrics_path = Path(tmp) / "bag_a" / "naive" / "metrics_per_frame.csv"
            _write_metrics(
                metrics_path,
                [
                    {
                        "bag_name": "bag_a",
                        "variant_name": "naive",
                        "frame_index": 0,
                        "pair_dt_sec": 0.01,
                        "pair_accepted": 1,
                        "num_input_points": 100,
                        "num_points_projected_in_image": 50,
                        "num_output_points": 45,
                        "output_retention_ratio": 0.9,
                        "runtime_total_ms": 10.0,
                        "runtime_projection_ms": 3.0,
                        "runtime_publish_ms": 1.0,
                        "num_would_hit_invalid_mask": 6,
                        "num_would_hit_depth_edge": 8,
                        "num_would_fail_occlusion": 2,
                        "projection_projected_ratio": 0.5,
                        "projection_in_front_ratio": 0.8,
                        "projection_in_image_ratio": 0.625,
                        "projection_invalid_mask_hit_ratio": 0.12,
                        "projection_confidence_rejection_ratio": 0.0,
                        "projection_depth_edge_rejection_ratio": 0.0,
                        "projection_occlusion_rejection_ratio": 0.0,
                        "projection_health_score": 0.88,
                        "num_projection_suppressed": 5,
                        "num_projection_rejected": 5,
                        "num_projection_accepted": 45,
                    },
                    {
                        "bag_name": "bag_a",
                        "variant_name": "naive",
                        "frame_index": 1,
                        "pair_dt_sec": 0.05,
                        "pair_accepted": 0,
                        "drop_reason": "pair_dt_too_large",
                        "runtime_total_ms": 20.0,
                    },
                ],
            )

            summary = summarize_metrics_file(metrics_path)

            self.assertEqual(summary["bag_name"], "bag_a")
            self.assertEqual(summary["variant_name"], "naive")
            self.assertEqual(summary["num_frames_seen"], 2)
            self.assertEqual(summary["num_frames_accepted"], 1)
            self.assertEqual(summary["pair_drop_rate"], 0.5)
            self.assertEqual(summary["mean_pair_dt_sec"], 0.03)
            self.assertEqual(summary["p95_runtime_total_ms"], 19.5)
            self.assertTrue((metrics_path.parent / "summary.csv").exists())
            with (metrics_path.parent / "summary.json").open(encoding="utf-8") as handle:
                summary_json = json.load(handle)
                self.assertEqual(summary_json["mean_output_points"], 45.0)
                self.assertEqual(summary_json["mean_projection_health_score"], 0.88)
                self.assertEqual(summary_json["mean_projection_projected_ratio"], 0.5)

    def test_write_results_tables_aggregates_variants(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            for variant, rejected in [("naive", 0), ("full", 5)]:
                metrics_path = tmp_path / "bag_a" / variant / "metrics_per_frame.csv"
                _write_metrics(
                    metrics_path,
                    [
                        {
                            "bag_name": "bag_a",
                            "variant_name": variant,
                            "frame_index": 0,
                            "pair_dt_sec": 0.01,
                            "pair_accepted": 1,
                            "num_input_points": 100,
                            "num_points_projected_in_image": 50,
                            "num_rejected_invalid_mask": rejected,
                            "num_rejected_depth_edge": rejected,
                            "num_rejected_occlusion": rejected,
                            "num_output_points": 50 - rejected,
                            "output_retention_ratio": (50 - rejected) / 50,
                            "runtime_total_ms": 10.0 + rejected,
                            "runtime_projection_ms": 3.0,
                            "runtime_mask_ms": 1.0,
                            "runtime_depth_edge_ms": 2.0,
                            "runtime_occlusion_ms": 1.0,
                            "runtime_publish_ms": 1.0,
                            "num_would_hit_invalid_mask": 6,
                            "num_would_hit_depth_edge": 8,
                            "num_would_fail_occlusion": 2,
                        }
                    ],
                )
                summarize_metrics_file(metrics_path)

            write_results_tables(tmp_path)

            ablation_path = tmp_path / "paper" / "table_ablation.csv"
            runtime_path = tmp_path / "paper" / "table_runtime.csv"
            bags_path = tmp_path / "paper" / "table_bags.csv"
            self.assertTrue(ablation_path.exists())
            self.assertTrue(runtime_path.exists())
            self.assertTrue(bags_path.exists())

            with ablation_path.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual([row["Method"] for row in rows], ["Naive", "Full"])
            self.assertEqual(rows[0]["Would-hit invalid mask/frame"], "6.000")
            self.assertEqual(rows[1]["Invalid-mask rejected/frame"], "5.000")


if __name__ == "__main__":
    unittest.main()
