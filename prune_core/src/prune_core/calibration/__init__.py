#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Classical online calibration helpers (ROS-agnostic).

Exports
-------
- ``OnlineCalibrationHealth``:
  Lightweight health/uncertainty estimator for online calibration monitoring.
- ``CalibrationHealthSnapshot``:
  Dataclass carrying aligned diagnostics for logging, plotting, and ROS topics.

For the full methodology used in this repository, see:
``docs/manual/online_calibration_methodology.md``.
"""

from prune_core.calibration.online_health import (
    CalibrationHealthSnapshot,
    OnlineCalibrationHealth,
)

__all__ = ["CalibrationHealthSnapshot", "OnlineCalibrationHealth"]
