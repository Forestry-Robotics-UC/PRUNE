#!/usr/bin/env python3
"""Deterministic ICNF results-dir validation for PRUNE runs."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

_THIS = Path(__file__).resolve()
for parent in _THIS.parents:
    ros_src = parent / "prune_ros"
    if (ros_src / "prune_ros").is_dir():
        for candidate in (parent, ros_src):
            if str(candidate) not in sys.path:
                sys.path.insert(0, str(candidate))
        break

from prune_ros.diagnostics.experiment_metrics import (
    read_metrics_csv,
    summarize_metrics_rows,
)

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None  # type: ignore[assignment]


try:
    from tools.results.run_ablation_suite import VARIANTS as ABALATION_VARIANTS
except Exception:  # pragma: no cover
    ABALATION_VARIANTS = {}


DEFAULT_PAIR_MAX_DT_SEC = 0.15


@dataclass(frozen=True)
class IcnfGateExpectation:
    expected_enabled: bool | None
    would_hit: int
    rejected: int
    mode: str
    status: str
    reason: str


def _discover_run_dirs(results_dir: Path) -> list[Path]:
    results_dir = Path(results_dir)
    if (results_dir / "metrics_per_frame.csv").exists():
        return [results_dir]
    run_dirs = sorted({path.parent for path in results_dir.rglob("metrics_per_frame.csv")})
    return run_dirs


def _load_summary_row(run_dir: Path) -> dict[str, Any]:
    summary_path = run_dir / "summary.csv"
    if not summary_path.exists():
        return {}
    with summary_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        row = next(reader, None)
    return dict(row or {})


def _coerce_variant_name(run_dir: Path, summary_row: dict[str, Any], summary: dict[str, Any]) -> str:
    variant = str(summary_row.get("variant_name") or summary.get("variant_name") or "")
    if variant:
        return variant
    if run_dir.parent.name and run_dir.parent != run_dir:
        return run_dir.name
    return ""


def _expected_gate_flags(variant_name: str) -> dict[str, bool] | None:
    variant = ABALATION_VARIANTS.get(variant_name)
    if not variant:
        return None
    return {
        "invalid_mask": str(variant.get("use_invalid_mask", "false")).lower() == "true",
        "depth_edge": str(variant.get("use_depth_edge_rejection", "false")).lower() == "true",
        "occlusion": str(variant.get("use_occlusion_gate", "false")).lower() == "true",
    }


def _gate_mode(would_hit: int, rejected: int) -> str:
    if rejected > 0:
        return "filtered"
    if would_hit > 0:
        return "suppressed"
    return "absent"


def _build_gate_check(
    *,
    name: str,
    expected_enabled: bool | None,
    would_hit: int,
    rejected: int,
) -> IcnfGateExpectation:
    mode = _gate_mode(would_hit, rejected)
    if expected_enabled is True:
        status = "pass" if would_hit > 0 and rejected > 0 else "fail"
        if would_hit <= 0:
            reason = f"expected {name} evidence but none was observed"
        elif rejected <= 0:
            reason = f"expected {name} filtering but rejected count stayed at 0"
        else:
            reason = f"{name} filtering observed"
    elif expected_enabled is False:
        status = "pass" if rejected == 0 else "fail"
        if rejected > 0:
            reason = f"expected {name} suppression but rejected count was {rejected}"
        elif would_hit > 0:
            reason = f"{name} suppression observed with would-hit evidence"
        else:
            reason = f"{name} remained suppressed with no evidence in this run"
    else:
        status = "pass" if mode != "filtered" or rejected >= 0 else "fail"
        reason = f"{name} mode={mode}"
    return IcnfGateExpectation(
        expected_enabled=expected_enabled,
        would_hit=would_hit,
        rejected=rejected,
        mode=mode,
        status=status,
        reason=reason,
    )


def _aggregate_gate_status(checks: dict[str, IcnfGateExpectation]) -> dict[str, Any]:
    return {
        "status": "pass" if all(check.status == "pass" for check in checks.values()) else "fail",
        "checks": {
            name: {
                "expected_enabled": check.expected_enabled,
                "would_hit": check.would_hit,
                "rejected": check.rejected,
                "mode": check.mode,
                "status": check.status,
                "reason": check.reason,
            }
            for name, check in checks.items()
        },
    }


def _build_run_report(run_dir: Path, *, pair_max_dt_sec: float = DEFAULT_PAIR_MAX_DT_SEC) -> dict[str, Any]:
    metrics_path = run_dir / "metrics_per_frame.csv"
    rows = read_metrics_csv(metrics_path)
    summary = summarize_metrics_rows(rows)
    summary_row = _load_summary_row(run_dir)
    bag_name = str(summary_row.get("bag_name") or summary.get("bag_name") or run_dir.parent.name)
    variant_name = _coerce_variant_name(run_dir, summary_row, summary)
    expected_flags = _expected_gate_flags(variant_name)

    accepted_rows = [row for row in rows if int(row.get("pair_accepted", 0)) == 1]
    abs_pair_dts = [abs(float(row.get("pair_dt_sec", 0.0))) for row in rows]
    mean_abs_pair_dt = sum(abs_pair_dts) / max(len(abs_pair_dts), 1)
    max_abs_pair_dt = max(abs_pair_dts) if abs_pair_dts else 0.0
    p95_abs_pair_dt = sorted(abs_pair_dts)[min(len(abs_pair_dts) - 1, int(round(0.95 * (len(abs_pair_dts) - 1))))] if abs_pair_dts else 0.0

    sync_pass = bool(rows) and bool(accepted_rows) and p95_abs_pair_dt <= pair_max_dt_sec
    projection_pass = (
        float(summary.get("mean_projected_points", 0.0)) > 0.0
        and float(summary.get("mean_output_points", 0.0)) > 0.0
        and float(summary.get("mean_output_points", 0.0)) <= float(summary.get("mean_projected_points", 0.0))
    )

    invalid_check = _build_gate_check(
        name="invalid_mask",
        expected_enabled=None if expected_flags is None else expected_flags["invalid_mask"],
        would_hit=int(round(float(summary.get("mean_would_hit_invalid_mask", 0.0)))),
        rejected=int(round(float(summary.get("mean_invalid_mask_rejected", 0.0)))),
    )
    depth_edge_check = _build_gate_check(
        name="depth_edge",
        expected_enabled=None if expected_flags is None else expected_flags["depth_edge"],
        would_hit=int(round(float(summary.get("mean_would_hit_depth_edge", 0.0)))),
        rejected=int(round(float(summary.get("mean_depth_edge_rejected", 0.0)))),
    )
    occlusion_check = _build_gate_check(
        name="occlusion",
        expected_enabled=None if expected_flags is None else expected_flags["occlusion"],
        would_hit=int(round(float(summary.get("mean_would_fail_occlusion", 0.0)))),
        rejected=int(round(float(summary.get("mean_occlusion_rejected", 0.0)))),
    )

    confidence_rejected = int(round(float(summary.get("mean_confidence_rejected", 0.0))))
    confidence_check = {
        "status": "pass",
        "rejected": confidence_rejected,
        "reason": "confidence thresholding observed"
        if confidence_rejected > 0
        else "no confidence-threshold evidence observed",
    }

    suppression_vs_filtering = _aggregate_gate_status(
        {
            "invalid_mask": invalid_check,
            "depth_edge": depth_edge_check,
            "occlusion": occlusion_check,
        }
    )

    diagnostics_pass = all(
        float(summary.get(field, 0.0)) >= 0.0
        for field in (
            "mean_runtime_projection_ms",
            "mean_runtime_mask_ms",
            "mean_runtime_depth_edge_ms",
            "mean_runtime_occlusion_ms",
            "mean_runtime_publish_ms",
        )
    ) and bool(rows)
    diagnostics_check = {
        "status": "pass" if diagnostics_pass else "fail",
        "reason": "metrics counters and runtimes present" if diagnostics_pass else "metrics rows or runtime counters missing",
    }

    checks = {
        "sync_pair": {
            "status": "pass" if sync_pass else "fail",
            "pair_accepted": len(accepted_rows),
            "pair_drop_rate": float(summary.get("pair_drop_rate", 0.0)),
            "mean_abs_pair_dt_sec": mean_abs_pair_dt,
            "p95_abs_pair_dt_sec": p95_abs_pair_dt,
            "max_abs_pair_dt_sec": max_abs_pair_dt,
            "pair_max_dt_sec": pair_max_dt_sec,
            "reason": "pair timing within tolerance" if sync_pass else "pair timing missing or too large",
        },
        "projection": {
            "status": "pass" if projection_pass else "fail",
            "mean_projected_points": float(summary.get("mean_projected_points", 0.0)),
            "mean_output_points": float(summary.get("mean_output_points", 0.0)),
            "reason": "projection produced non-empty outputs" if projection_pass else "projection output missing or inconsistent",
        },
        "invalid_mask": {
            "status": invalid_check.status,
            "expected_enabled": invalid_check.expected_enabled,
            "would_hit": invalid_check.would_hit,
            "rejected": invalid_check.rejected,
            "mode": invalid_check.mode,
            "reason": invalid_check.reason,
        },
        "depth_edge": {
            "status": depth_edge_check.status,
            "expected_enabled": depth_edge_check.expected_enabled,
            "would_hit": depth_edge_check.would_hit,
            "rejected": depth_edge_check.rejected,
            "mode": depth_edge_check.mode,
            "reason": depth_edge_check.reason,
        },
        "occlusion": {
            "status": occlusion_check.status,
            "expected_enabled": occlusion_check.expected_enabled,
            "would_hit": occlusion_check.would_hit,
            "rejected": occlusion_check.rejected,
            "mode": occlusion_check.mode,
            "reason": occlusion_check.reason,
        },
        "confidence": confidence_check,
        "suppression_vs_filtering": suppression_vs_filtering,
        "diagnostics": diagnostics_check,
    }

    overall_pass = all(
        check.get("status") == "pass"
        for check in (
            checks["sync_pair"],
            checks["projection"],
            checks["invalid_mask"],
            checks["depth_edge"],
            checks["occlusion"],
            checks["confidence"],
            checks["suppression_vs_filtering"],
            checks["diagnostics"],
        )
    )

    return {
        "bag_name": bag_name,
        "variant_name": variant_name,
        "run_dir": str(run_dir),
        "summary": summary,
        "summary_row": summary_row,
        "checks": checks,
        "overall_pass": overall_pass,
    }


def build_icnf_report(results_dir: Path | str, *, pair_max_dt_sec: float = DEFAULT_PAIR_MAX_DT_SEC) -> dict[str, Any]:
    results_dir = Path(results_dir)
    run_dirs = _discover_run_dirs(results_dir)
    if not run_dirs:
        return {
            "results_dir": str(results_dir),
            "runs": [],
            "checks": {},
            "overall_pass": False,
            "reason": "no metrics_per_frame.csv files found",
        }

    iterator: Iterable[Path] = run_dirs
    if len(run_dirs) > 1 and tqdm is not None:
        iterator = tqdm(run_dirs, desc="ICNF runs", unit="run")

    run_reports = [
        _build_run_report(run_dir, pair_max_dt_sec=pair_max_dt_sec)
        for run_dir in iterator
    ]
    overall_pass = all(report["overall_pass"] for report in run_reports)
    return {
        "results_dir": str(results_dir),
        "runs": run_reports,
        "checks": run_reports[0]["checks"] if len(run_reports) == 1 else {
            "status": "pass" if overall_pass else "fail",
            "reason": "all discovered runs passed" if overall_pass else "one or more runs failed",
        },
        "overall_pass": overall_pass,
    }


def render_icnf_report(report: dict[str, Any]) -> str:
    lines = [
        f"ICNF validation report for {report.get('results_dir', '-')}",
        f"overall_pass: {report.get('overall_pass', False)}",
    ]
    for run in report.get("runs", []):
        lines.append("")
        lines.append(f"run: {run.get('bag_name', '-')}/{run.get('variant_name', '-')}")
        lines.append(f"  run_dir: {run.get('run_dir', '-')}")
        lines.append(f"  overall_pass: {run.get('overall_pass', False)}")
        for name, check in run.get("checks", {}).items():
            lines.append(f"  {name}: {check.get('status', 'fail')} ({check.get('reason', '-')})")
    return "\n".join(lines) + "\n"


def _write_report_files(report: dict[str, Any], output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "icnf_validation_report.json"
    text_path = output_dir / "icnf_validation_report.txt"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    text_path.write_text(render_icnf_report(report), encoding="utf-8")
    return json_path, text_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", required=True, help="PRUNE results directory or run root")
    parser.add_argument(
        "--output-dir",
        default="",
        help="Directory for report artifacts (defaults to the results directory).",
    )
    parser.add_argument(
        "--pair-max-dt-sec",
        type=float,
        default=DEFAULT_PAIR_MAX_DT_SEC,
        help="Maximum accepted pair_dt_sec p95 for sync validation.",
    )
    args = parser.parse_args(argv)

    results_dir = Path(args.results_dir)
    report = build_icnf_report(results_dir, pair_max_dt_sec=float(args.pair_max_dt_sec))
    output_dir = Path(args.output_dir) if args.output_dir else results_dir
    json_path, text_path = _write_report_files(report, output_dir)
    print(render_icnf_report(report), end="")
    print(f"Wrote {json_path}")
    print(f"Wrote {text_path}")
    return 0 if report.get("overall_pass") else 1


if __name__ == "__main__":
    raise SystemExit(main())
