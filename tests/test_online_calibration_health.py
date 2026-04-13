#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Derived from Semantic SLAM
#
# Original Author:
#   Xuan Zhang
#
# Subsequent Contributions:
#   David Russell
#
# Modified by:
#   Duda Andrada (ENTFAC Sensor Fusion)
#
# Original project:
#   https://github.com/floatlazer/semantic_slam
#
# Author: Duda Andrada
# Maintainer: Duda Andrada <duda.andrada@isr.uc.pt>
# License: GNU General Public License v3.0 (GPL-3.0)
# Repository: ENTFAC-Sensor-Fusion
#
# Description:
#   Unit tests for lightweight online calibration-health estimator.

import sys
from pathlib import Path

import numpy as np

CORE_SRC = Path(__file__).resolve().parents[1] / "entfac_fusion_core" / "src"
if str(CORE_SRC) not in sys.path:
    sys.path.insert(0, str(CORE_SRC))

from entfac_fusion_core.calibration import OnlineCalibrationHealth  # noqa: E402


def test_online_health_snapshot_ranges_are_clipped():
    tracker = OnlineCalibrationHealth(
        ema_alpha=0.2,
        std_window=10,
        std_scale=0.1,
        score_center=0.2,
        score_scale=0.1,
        min_observability=0.1,
    )
    snap = tracker.update(
        score_raw=0.5,
        observability=0.9,
        correction_rpy_rad=np.array([0.0, 0.0, 0.0], dtype=float),
        correction_uncertainty=0.2,
    )
    assert 0.0 <= snap.health <= 1.0
    assert 0.0 <= snap.confidence <= 1.0
    assert 0.0 <= snap.uncertainty <= 1.0


def test_online_health_increases_with_better_alignment_and_observability():
    tracker = OnlineCalibrationHealth(
        ema_alpha=1.0,
        std_window=8,
        std_scale=0.2,
        score_center=0.25,
        score_scale=0.1,
        min_observability=0.1,
    )
    low = tracker.update(
        score_raw=0.05,
        observability=0.2,
        correction_rpy_rad=np.zeros(3, dtype=float),
        correction_uncertainty=0.8,
    )
    high = tracker.update(
        score_raw=0.55,
        observability=0.9,
        correction_rpy_rad=np.zeros(3, dtype=float),
        correction_uncertainty=0.2,
    )
    assert high.health > low.health
    assert high.uncertainty < low.uncertainty


def test_online_health_observability_gate_disables_confidence_when_too_low():
    tracker = OnlineCalibrationHealth(
        ema_alpha=1.0,
        std_window=8,
        std_scale=0.2,
        score_center=0.2,
        score_scale=0.1,
        min_observability=0.4,
    )
    snap = tracker.update(
        score_raw=0.9,
        observability=0.1,
        correction_rpy_rad=np.zeros(3, dtype=float),
        correction_uncertainty=0.5,
    )
    assert snap.confidence == 0.0
    assert snap.health == 0.0
    assert snap.uncertainty >= 0.5


def test_online_health_reports_correction_angles_in_degrees():
    tracker = OnlineCalibrationHealth(
        ema_alpha=0.3,
        std_window=10,
        std_scale=0.1,
        score_center=0.2,
        score_scale=0.1,
        min_observability=0.1,
    )
    snap = tracker.update(
        score_raw=0.4,
        observability=0.8,
        correction_rpy_rad=np.deg2rad(np.array([1.0, -2.0, 3.0], dtype=float)),
        correction_uncertainty=0.3,
    )
    assert np.isclose(snap.correction_roll_deg, 1.0, atol=1e-6)
    assert np.isclose(snap.correction_pitch_deg, -2.0, atol=1e-6)
    assert np.isclose(snap.correction_yaw_deg, 3.0, atol=1e-6)
