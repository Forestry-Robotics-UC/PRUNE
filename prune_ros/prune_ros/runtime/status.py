#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Author: Duda Andrada
# Maintainer: Duda Andrada <duda.andrada@isr.uc.pt>
# License: GNU General Public License v3.0 (GPL-3.0)
# Repository: PRUNE
#
# Description:
#   Periodic, low-noise status reporting (publish rate, point counts) for ROS logs.

"""Periodic status reporting for ROS nodes."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple


@dataclass
class StatusSnapshot:
    mode: str
    semantic_input_type: str
    points_last: int
    pub_hz: float
    avg_points: float
    avg_callback_ms: float


class StatusReporter:
    """Collect publish stats and print a compact ASCII table periodically."""

    def __init__(self, *, period_sec: float):
        self._period_sec = float(period_sec)
        self._t_start = None
        self._t_last = None
        self._count = 0
        self._points_sum = 0
        self._points_last = 0
        self._cb_time_sum = 0.0

    @property
    def enabled(self) -> bool:
        return self._period_sec > 0.0

    def record(self, *, points: int, callback_sec: float = 0.0) -> Optional[StatusSnapshot]:
        if not self.enabled:
            return None

        now = time.time()
        if self._t_start is None:
            self._t_start = now
            self._t_last = now

        self._count += 1
        self._points_sum += int(points)
        self._points_last = int(points)
        self._cb_time_sum += float(callback_sec)

        dt = float(now - self._t_last)
        if dt < self._period_sec:
            return None

        pub_hz = float(self._count) / dt if dt > 0.0 else 0.0
        avg_points = float(self._points_sum) / float(self._count) if self._count else 0.0
        avg_cb_ms = (
            1000.0 * float(self._cb_time_sum) / float(self._count) if self._count else 0.0
        )

        snapshot = StatusSnapshot(
            mode="",
            semantic_input_type="",
            points_last=self._points_last,
            pub_hz=pub_hz,
            avg_points=avg_points,
            avg_callback_ms=avg_cb_ms,
        )

        self._t_last = now
        self._count = 0
        self._points_sum = 0
        self._cb_time_sum = 0.0
        return snapshot


def render_kv_table(rows: Sequence[Tuple[str, str]]) -> str:
    """Render a compact ASCII key-value table.

    Values may contain newlines; in that case, subsequent lines are rendered with
    an empty key column.
    """
    expanded: List[Tuple[str, str]] = []
    for key, value in rows:
        key_str = str(key)
        value_lines = str(value).splitlines() or [""]
        for idx, line in enumerate(value_lines):
            expanded.append((key_str if idx == 0 else "", line))

    key_w = max((len(k) for k, _ in expanded), default=1)
    val_w = max((len(v) for _, v in expanded), default=1)
    horiz = "+" + "-" * (key_w + 2) + "+" + "-" * (val_w + 2) + "+"
    lines = [horiz]
    for k, v in expanded:
        lines.append(f"| {k:<{key_w}} | {v:<{val_w}} |")
    lines.append(horiz)
    return "\n".join(lines)


def render_status_table(
    *,
    node_name: str,
    mode: str,
    semantic_input_type: str,
    target_frame: str,
    output_topic: str,
    points_last: int,
    pub_hz: float,
    avg_points: float,
    avg_callback_ms: float,
) -> str:
    name = str(node_name).lstrip("/")
    mode_s = str(mode)
    sem_s = str(semantic_input_type)
    target_s = str(target_frame)
    out_s = str(output_topic)
    rows = [
        ("node", name),
        ("mode", mode_s),
        ("semantic", sem_s),
        ("target", target_s),
        ("output", out_s),
        ("pub_hz", f"{pub_hz:.2f}"),
        ("points_last", str(int(points_last))),
        ("points_avg", f"{avg_points:.1f}"),
        ("cb_ms_avg", f"{avg_callback_ms:.2f}"),
    ]
    return render_kv_table(rows)
