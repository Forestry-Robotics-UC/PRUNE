"""Synchronization-related prune parameter loaders."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .config_helpers import get_float, get_int


@dataclass
class SyncConfig:
    sync_slop_sec: float
    pair_max_dt_sec: float
    semantic_time_offset_sec: float
    sync_queue_size: int
    cloud_time_offset_sec: float
    cloud_stamp_source: str
    stamp_debug_log_period_sec: float


def load_sync_config(node: Any) -> SyncConfig:
    cloud_stamp_source = node._get_param_str('~cloud_stamp_source', '', 'Timestamp source for published PointCloud2: auto, semantic, depth, lidar, latest, earliest, midpoint.', allow_empty=True)
    return SyncConfig(
        sync_slop_sec=get_float(node, '~sync_slop_sec', 0.1, 'ApproximateTimeSynchronizer slop in seconds for semantic/depth or semantic/lidar pairing.', min_value=0.0),
        pair_max_dt_sec=get_float(node, '~pair_max_dt_sec', 0.03, 'Hard max allowed |Δt| (seconds) between semantic and geometry; <=0 disables.', min_value=0.0),
        semantic_time_offset_sec=node._get_param_float('~semantic_time_offset_sec', 0.0, 'Signed offset (seconds) applied to semantic timestamps for pairing and timestamp selection (negative shifts semantic earlier).'),
        sync_queue_size=get_int(node, '~sync_queue_size', 5, 'ApproximateTimeSynchronizer queue size for semantic/depth or semantic/lidar pairing.', min_value=1),
        cloud_time_offset_sec=node._get_param_float('~cloud_time_offset_sec', 0.0, 'Signed offset (seconds) added to published cloud timestamps (negative shifts earlier).'),
        cloud_stamp_source=(cloud_stamp_source or '').strip().lower(),
        stamp_debug_log_period_sec=get_float(node, '~stamp_debug_log_period_sec', 2.0, 'Minimum period (seconds) between timestamp/offset debug logs; set 0 to log every callback when debug=true.', min_value=0.0),
    )
