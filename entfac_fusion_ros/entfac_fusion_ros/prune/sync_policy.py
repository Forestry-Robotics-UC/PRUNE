"""Timestamp and synchronization policy helpers for prune fusion."""

from __future__ import annotations

import time
from typing import Any, Optional, Tuple

import rospy

from .config import SyncConfig


class StampPolicy:
    """Resolve cloud stamps and validate semantic-geometry pairing."""

    def __init__(self, node: Any, config: SyncConfig, logger: Any):
        self._node = node
        self._config = config
        self._log = logger

    @property
    def cloud_stamp_source(self) -> str:
        return getattr(self._node, "cloud_stamp_source", "")

    def resolve_cloud_stamp_source(self) -> None:
        raw = (self.cloud_stamp_source or "").strip().lower()
        if not raw or raw == "auto":
            resolved = "latest" if self._node.mode == "depth" else "semantic"
        else:
            aliases = {
                "semantic": "semantic",
                "sem": "semantic",
                "labels": "semantic",
                "label": "semantic",
                "max": "latest",
                "latest": "latest",
                "min": "earliest",
                "earliest": "earliest",
                "avg": "midpoint",
                "average": "midpoint",
                "mid": "midpoint",
                "middle": "midpoint",
                "midpoint": "midpoint",
                "depth": "depth",
                "lidar": "lidar",
            }
            resolved = aliases.get(raw)
            if resolved is None:
                self._log.warn(
                    "_resolve_cloud_stamp_source",
                    "Unknown ~cloud_stamp_source=%r; falling back to auto.",
                    raw,
                )
                resolved = "latest" if self._node.mode == "depth" else "semantic"

        valid = {
            "depth": {"semantic", "depth", "latest", "earliest", "midpoint"},
            "lidar": {"semantic", "lidar", "latest", "earliest", "midpoint"},
        }
        if resolved not in valid.get(self._node.mode, set()):
            self._log.warn(
                "_resolve_cloud_stamp_source",
                "Invalid ~cloud_stamp_source=%s for mode=%s; falling back to auto.",
                resolved,
                self._node.mode,
            )
            resolved = "latest" if self._node.mode == "depth" else "semantic"

        self._node.cloud_stamp_source = resolved
        if "~cloud_stamp_source" in getattr(self._node, "_param_meta", {}):
            self._node._param_meta["~cloud_stamp_source"]["value"] = resolved

    def validate_depth_pair(self, sem_msg, depth_msg) -> Optional[Tuple[rospy.Time, rospy.Time]]:
        sem_stamp = sem_msg.header.stamp
        depth_stamp = depth_msg.header.stamp
        sem_pair_stamp = self.apply_stamp_offset(sem_stamp, self._node.semantic_time_offset_sec)
        if self._node.pair_max_dt_sec > 0.0:
            dt = abs((sem_pair_stamp - depth_stamp).to_sec())
            if dt > self._node.pair_max_dt_sec:
                self._log.warn(
                    "_depth_callback",
                    "Dropping pair: |Δt|=%.6fs > %.6fs",
                    dt,
                    float(self._node.pair_max_dt_sec),
                )
                return None
        chosen_stamp = self.choose_cloud_stamp(sem_pair_stamp, depth_stamp, "depth")
        stamp = self.apply_cloud_time_offset(chosen_stamp)
        self.log_stamp_debug(
            "_depth_callback",
            sem_stamp,
            depth_stamp,
            "depth",
            chosen_stamp,
            stamp,
            sem_pair_stamp=sem_pair_stamp,
        )
        return chosen_stamp, stamp

    def validate_lidar_pair(self, sem_msg, lidar_msg) -> Optional[Tuple[rospy.Time, rospy.Time]]:
        sem_stamp = sem_msg.header.stamp
        lidar_stamp = lidar_msg.header.stamp
        sem_pair_stamp = self.apply_stamp_offset(sem_stamp, self._node.semantic_time_offset_sec)
        if self._node.pair_max_dt_sec > 0.0:
            dt = abs((sem_pair_stamp - lidar_stamp).to_sec())
            if dt > self._node.pair_max_dt_sec:
                self._log.warn(
                    "_lidar_callback",
                    "Dropping pair: |Δt|=%.6fs > %.6fs",
                    dt,
                    float(self._node.pair_max_dt_sec),
                )
                return None
        chosen_stamp = self.choose_cloud_stamp(sem_pair_stamp, lidar_stamp, "lidar")
        stamp = self.apply_cloud_time_offset(chosen_stamp)
        self.log_stamp_debug(
            "_lidar_callback",
            sem_stamp,
            lidar_stamp,
            "lidar",
            chosen_stamp,
            stamp,
            sem_pair_stamp=sem_pair_stamp,
        )
        return chosen_stamp, stamp

    def compute_pair_dt_sec(self, sem_msg, lidar_msg) -> float:
        sem_stamp = sem_msg.header.stamp
        lidar_stamp = lidar_msg.header.stamp
        sem_pair_stamp = self.apply_stamp_offset(sem_stamp, self._node.semantic_time_offset_sec)
        return abs((sem_pair_stamp - lidar_stamp).to_sec())

    def apply_cloud_time_offset(self, stamp: rospy.Time) -> rospy.Time:
        if self._node.cloud_time_offset_sec == 0.0 or stamp == rospy.Time():
            return stamp
        shifted = stamp + rospy.Duration(self._node.cloud_time_offset_sec)
        if shifted.to_sec() < 0.0:
            return rospy.Time(0)
        return shifted

    def apply_stamp_offset(self, stamp: rospy.Time, offset_sec: float) -> rospy.Time:
        if stamp == rospy.Time() or offset_sec == 0.0:
            return stamp
        return stamp + rospy.Duration(offset_sec)

    def choose_cloud_stamp(
        self,
        sem_stamp: rospy.Time,
        other_stamp: rospy.Time,
        other_label: str,
    ) -> rospy.Time:
        if sem_stamp == rospy.Time():
            return other_stamp
        if other_stamp == rospy.Time():
            return sem_stamp

        source = self.cloud_stamp_source
        if source == "semantic":
            return sem_stamp
        if source == other_label:
            return other_stamp
        if source == "latest":
            return sem_stamp if sem_stamp > other_stamp else other_stamp
        if source == "earliest":
            return sem_stamp if sem_stamp < other_stamp else other_stamp
        if source == "midpoint":
            mid_sec = 0.5 * (sem_stamp.to_sec() + other_stamp.to_sec())
            return rospy.Time.from_sec(mid_sec)

        return sem_stamp if sem_stamp > other_stamp else other_stamp

    def log_stamp_debug(
        self,
        context: str,
        sem_stamp: rospy.Time,
        other_stamp: rospy.Time,
        other_label: str,
        chosen_stamp: rospy.Time,
        shifted_stamp: rospy.Time,
        sem_pair_stamp: Optional[rospy.Time] = None,
    ) -> None:
        if not self._node.debug:
            return
        now = time.time()
        period = float(self._node.stamp_debug_log_period_sec)
        if period > 0.0 and (now - self._node._stamp_debug_last_log_at) < period:
            return
        self._node._stamp_debug_last_log_at = now
        sem_pair_stamp = sem_stamp if sem_pair_stamp is None else sem_pair_stamp
        raw_dt_sec = (sem_stamp - other_stamp).to_sec()
        pair_dt_sec = (sem_pair_stamp - other_stamp).to_sec()
        self._log.debug(
            context,
            "stamps: semantic=%.9f semantic_pair=%.9f %s=%.9f raw_dt=%.9f pair_dt=%.9f chosen=%.9f shifted=%.9f source=%s semantic_offset=%.6f cloud_offset=%.6f",
            sem_stamp.to_sec(),
            sem_pair_stamp.to_sec(),
            other_label,
            other_stamp.to_sec(),
            raw_dt_sec,
            pair_dt_sec,
            chosen_stamp.to_sec(),
            shifted_stamp.to_sec(),
            self.cloud_stamp_source,
            float(self._node.semantic_time_offset_sec),
            float(self._node.cloud_time_offset_sec),
        )
