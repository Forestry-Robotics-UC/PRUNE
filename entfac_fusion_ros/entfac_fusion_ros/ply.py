#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# ENTFAC Sensor Fusion implementation.
#
# Note:
#   This file was developed specifically for ENTFAC Sensor Fusion.
#   Project-level upstream attribution is documented in README.md.
#
# Modified by:
#   Duda Andrada (ENTFAC Sensor Fusion)
#
# Author: Duda Andrada
# Maintainer: Duda Andrada <duda.andrada@isr.uc.pt>
# License: GNU General Public License v3.0 (GPL-3.0)
# Repository: ENTFAC-Sensor-Fusion
#
# Description:
#   Binary little-endian PLY writer for semantic point clouds (xyz + label + rgb + confidence).

"""PLY I/O utilities."""

from __future__ import annotations

import queue
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import numpy as np

from entfac_fusion_ros.pc2 import labels_to_uint16


def _packed_u32_to_rgb_u8(packed_u32: np.ndarray) -> np.ndarray:
    packed = np.asarray(packed_u32, dtype=np.uint32).reshape(-1)
    r = ((packed >> 16) & 0xFF).astype(np.uint8, copy=False)
    g = ((packed >> 8) & 0xFF).astype(np.uint8, copy=False)
    b = (packed & 0xFF).astype(np.uint8, copy=False)
    return np.stack((r, g, b), axis=1)


def write_ply(
    path: Union[str, Path],
    points_xyz: np.ndarray,
    *,
    labels: Optional[np.ndarray] = None,
    confidence: Optional[np.ndarray] = None,
    rgb_packed_float: Optional[np.ndarray] = None,
) -> Path:
    """Write a binary_little_endian PLY file."""
    path = Path(path)
    points = np.asarray(points_xyz, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points_xyz must be (N, 3)")
    n = int(points.shape[0])

    dtype_fields = [("x", "<f4"), ("y", "<f4"), ("z", "<f4")]
    header = [
        "ply",
        "format binary_little_endian 1.0",
        f"element vertex {n}",
        "property float x",
        "property float y",
        "property float z",
    ]

    labels_u16 = None
    if labels is not None:
        labels_u16 = labels_to_uint16(labels)
        if labels_u16.shape[0] != n:
            raise ValueError("labels must be (N,) and aligned with points_xyz")
        dtype_fields.append(("label", "<u2"))
        header.append("property ushort label")

    conf_f32 = None
    if confidence is not None:
        conf_f32 = np.asarray(confidence, dtype=np.float32).reshape(-1)
        if conf_f32.shape[0] != n:
            raise ValueError("confidence must be (N,) and aligned with points_xyz")
        dtype_fields.append(("confidence", "<f4"))
        header.append("property float confidence")

    rgb_u8 = None
    if rgb_packed_float is not None:
        rgb_f = np.asarray(rgb_packed_float, dtype=np.float32).reshape(-1)
        if rgb_f.shape[0] != n:
            raise ValueError("rgb_packed_float must be (N,) and aligned with points_xyz")
        packed_u32 = rgb_f.astype("<f4", copy=False).view("<u4")
        rgb_u8 = _packed_u32_to_rgb_u8(packed_u32)
        dtype_fields.extend([("red", "u1"), ("green", "u1"), ("blue", "u1")])
        header.extend(
            [
                "property uchar red",
                "property uchar green",
                "property uchar blue",
            ]
        )

    header.append("end_header")
    dtype = np.dtype(dtype_fields)
    data = np.empty(n, dtype=dtype)
    data["x"] = points[:, 0]
    data["y"] = points[:, 1]
    data["z"] = points[:, 2]
    if labels_u16 is not None:
        data["label"] = labels_u16
    if conf_f32 is not None:
        data["confidence"] = conf_f32
    if rgb_u8 is not None:
        data["red"] = rgb_u8[:, 0]
        data["green"] = rgb_u8[:, 1]
        data["blue"] = rgb_u8[:, 2]

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as fh:
        fh.write(("\n".join(header) + "\n").encode("ascii"))
        fh.write(data.tobytes())
    return path


@dataclass(frozen=True)
class PlyJob:
    path: Path
    points_xyz: np.ndarray
    labels: Optional[np.ndarray]
    confidence: Optional[np.ndarray]
    rgb_packed_float: Optional[np.ndarray]


class PlyWriterThread:
    """Background writer to avoid blocking high-rate callbacks."""

    def __init__(self, *, queue_size: int = 2):
        self._queue: "queue.Queue[PlyJob]" = queue.Queue(maxsize=int(queue_size))
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._stop = threading.Event()
        self._started = False

    def start(self) -> None:
        if self._started:
            return
        self._started = True
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def enqueue(self, job: PlyJob) -> bool:
        try:
            self._queue.put_nowait(job)
            return True
        except queue.Full:
            return False

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                job = self._queue.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                write_ply(
                    job.path,
                    job.points_xyz,
                    labels=job.labels,
                    confidence=job.confidence,
                    rgb_packed_float=job.rgb_packed_float,
                )
            finally:
                self._queue.task_done()
