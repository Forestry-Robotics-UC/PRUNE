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
#   Lightweight online calibration-health estimator for LiDAR-camera fusion.

"""Classical online calibration-health estimator (ROS-agnostic).

This module provides a lightweight, online "calibration health head" designed for
edge deployment (CPU-first, no neural network assumptions). The estimator is
intended to be paired with a low-rate, bounded correction loop in the ROS node.

Method summary
--------------
For each processed frame, an alignment evidence score ``s_t`` (provided by the
caller) and an observability proxy ``o_t`` are fused into:

1. Smoothed alignment quality:

   ``s_ema(t) = (1 - alpha) * s_ema(t-1) + alpha * s_t``

2. Temporal stability:

   ``sigma_t = std({s_{t-k}} over sliding window)``

3. Confidence (gated by observability):

   ``confidence_t = exp(-sigma_t / sigma_scale) * gate(o_t)``

4. Health score:

   ``health_t = sigmoid((s_ema(t) - score_center) / score_scale) * confidence_t``

5. Uncertainty (conservative):

   ``uncertainty_t = max(1 - confidence_t, correction_uncertainty_t)``

The output includes health, uncertainty, and correction angles in degrees so ROS
wrappers can publish interpretable diagnostics.

Design rationale
----------------
- Exponential smoothing gives a robust low-pass estimate under short bursts.
- Sliding-window dispersion approximates confidence degradation in unstable scenes.
- Observability gating avoids overconfident outputs in weak geometry/texture.
- The class is deterministic, allocation-light, and independent of ROS.

References
----------
- Xia et al., WACV 2025, robust perception under sensor misalignment
  (uses misalignment uncertainty and temporal fusion ideas at system level).
  https://openaccess.thecvf.com/content/WACV2025/html/Xia_Robust_Long-Range_Perception_Against_Sensor_Misalignment_in_Autonomous_Vehicles_WACV_2025_paper.html
- LiDAR-camera calibration observability discussion:
  https://arxiv.org/abs/1903.06141
- Observability-aware calibration perspective:
  https://arxiv.org/abs/2205.03276
- Classical nonlinear observability rank condition:
  Hermann and Krener, IEEE TAC 1977, doi:10.1109/TAC.1977.1101601
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Optional

import numpy as np


def _sigmoid(x: float) -> float:
    x = float(np.clip(x, -60.0, 60.0))
    return float(1.0 / (1.0 + np.exp(-x)))


@dataclass
class CalibrationHealthSnapshot:
    """Single online calibration-health snapshot.

    Fields
    ------
    score_raw:
        Instantaneous alignment score passed by the caller.
    score_ema:
        Exponentially smoothed alignment score.
    score_std:
        Sliding-window standard deviation of raw score.
    observability:
        Scene observability proxy in [0, 1].
    confidence:
        Composite confidence from stability and observability gating.
    health:
        Final health score in [0, 1].
    uncertainty:
        Final uncertainty in [0, 1], conservative by construction.
    correction_uncertainty:
        Uncertainty provided by the correction stage (if any).
    correction_roll_deg / correction_pitch_deg / correction_yaw_deg:
        Current correction estimate converted to degrees for diagnostics.
    """

    score_raw: float
    score_ema: float
    score_std: float
    observability: float
    confidence: float
    health: float
    uncertainty: float
    correction_uncertainty: float
    correction_roll_deg: float
    correction_pitch_deg: float
    correction_yaw_deg: float


class OnlineCalibrationHealth:
    """Track online calibration health from alignment score + observability.

    This class is intentionally simple and edge-friendly:
    - Exponential moving average for robustness to short-term noise.
    - Sliding-window score std for temporal stability.
    - Observability gate to avoid overconfident estimates on weak scenes.
    """

    def __init__(
        self,
        *,
        ema_alpha: float = 0.15,
        std_window: int = 40,
        std_scale: float = 0.08,
        score_center: float = 0.25,
        score_scale: float = 0.10,
        min_observability: float = 0.15,
    ):
        ema_alpha = float(ema_alpha)
        std_window = int(std_window)
        std_scale = float(std_scale)
        score_scale = float(score_scale)
        min_observability = float(min_observability)
        if not 0.0 < ema_alpha <= 1.0:
            raise ValueError("ema_alpha must be in (0, 1]")
        if std_window < 2:
            raise ValueError("std_window must be >= 2")
        if std_scale <= 0.0:
            raise ValueError("std_scale must be > 0")
        if score_scale <= 0.0:
            raise ValueError("score_scale must be > 0")
        if not 0.0 <= min_observability <= 1.0:
            raise ValueError("min_observability must be in [0, 1]")

        self._ema_alpha = ema_alpha
        self._std_window = std_window
        self._std_scale = std_scale
        self._score_center = float(score_center)
        self._score_scale = score_scale
        self._min_observability = min_observability

        self._score_ema: Optional[float] = None
        self._score_history: Deque[float] = deque(maxlen=std_window)
        self._last: Optional[CalibrationHealthSnapshot] = None

    @property
    def last_snapshot(self) -> Optional[CalibrationHealthSnapshot]:
        return self._last

    def reset(self) -> None:
        self._score_ema = None
        self._score_history.clear()
        self._last = None

    def update(
        self,
        *,
        score_raw: float,
        observability: float,
        correction_rpy_rad: Optional[np.ndarray] = None,
        correction_uncertainty: float = 1.0,
    ) -> CalibrationHealthSnapshot:
        """Update health state from new alignment/observability evidence.

        Parameters
        ----------
        score_raw:
            Instantaneous alignment score (expected range is typically [-1, 1] for
            cosine/NCC-like scores, but any finite scalar is accepted).
        observability:
            Scene observability proxy in [0, 1].
        correction_rpy_rad:
            Optional correction estimate (roll, pitch, yaw) in radians.
        correction_uncertainty:
            Optional uncertainty from the correction estimator in [0, 1].

        Returns
        -------
        CalibrationHealthSnapshot
            Current calibrated-health snapshot.
        """
        score_raw = float(np.nan_to_num(score_raw, nan=0.0, posinf=0.0, neginf=0.0))
        observability = float(
            np.clip(
                np.nan_to_num(observability, nan=0.0, posinf=1.0, neginf=0.0),
                0.0,
                1.0,
            )
        )
        correction_uncertainty = float(
            np.clip(
                np.nan_to_num(correction_uncertainty, nan=1.0, posinf=1.0, neginf=1.0),
                0.0,
                1.0,
            )
        )

        if self._score_ema is None:
            score_ema = score_raw
        else:
            score_ema = (1.0 - self._ema_alpha) * self._score_ema + self._ema_alpha * score_raw
        self._score_ema = float(score_ema)
        self._score_history.append(float(score_raw))
        if len(self._score_history) > 1:
            score_std = float(np.std(np.asarray(self._score_history, dtype=np.float64)))
        else:
            score_std = 0.0

        alignment_term = _sigmoid((score_ema - self._score_center) / self._score_scale)
        stability_term = float(np.exp(-score_std / self._std_scale))
        observability_gate = 0.0
        if observability >= self._min_observability:
            denom = max(1e-6, 1.0 - self._min_observability)
            observability_gate = float(np.clip((observability - self._min_observability) / denom, 0.0, 1.0))
        confidence = float(np.clip(stability_term * observability_gate, 0.0, 1.0))
        health = float(np.clip(alignment_term * confidence, 0.0, 1.0))
        uncertainty = float(np.clip(max(1.0 - confidence, correction_uncertainty), 0.0, 1.0))

        if correction_rpy_rad is None:
            correction_rpy_rad = np.zeros(3, dtype=np.float64)
        else:
            correction_rpy_rad = np.asarray(correction_rpy_rad, dtype=np.float64).reshape(-1)
            if correction_rpy_rad.size != 3:
                raise ValueError(
                    "correction_rpy_rad must be shape (3,), got "
                    f"{correction_rpy_rad.shape}"
                )
        correction_deg = np.degrees(correction_rpy_rad)

        snap = CalibrationHealthSnapshot(
            score_raw=float(score_raw),
            score_ema=float(score_ema),
            score_std=float(score_std),
            observability=float(observability),
            confidence=float(confidence),
            health=float(health),
            uncertainty=float(uncertainty),
            correction_uncertainty=float(correction_uncertainty),
            correction_roll_deg=float(correction_deg[0]),
            correction_pitch_deg=float(correction_deg[1]),
            correction_yaw_deg=float(correction_deg[2]),
        )
        self._last = snap
        return snap
