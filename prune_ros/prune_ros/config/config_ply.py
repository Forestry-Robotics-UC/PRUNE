"""PLY export configuration for prune."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config_helpers import get_float


@dataclass
class PlyConfig:
    ply_output_dir: str
    ply_recording_enable: bool
    ply_target_frame: str
    ply_tf_use_latest: bool
    ply_tf_tolerance_sec: float


def load_ply_config(node: Any) -> PlyConfig:
    config = PlyConfig(
        ply_output_dir=node._get_param_str('~ply_output_dir', '', 'Optional directory for saved PLY files. Empty uses the current debug/output directory.', allow_empty=True),
        ply_recording_enable=node._get_param_bool('~ply_recording_enable', False, 'If true, write every published cloud to PLY asynchronously.'),
        ply_target_frame=node._get_param_str('~ply_target_frame', '', 'Optional target frame for saved PLY clouds. Empty uses the published cloud frame.', allow_empty=True),
        ply_tf_use_latest=node._get_param_bool('~ply_tf_use_latest', False, 'When true, PLY export uses the latest TF instead of exact timestamp lookup.'),
        ply_tf_tolerance_sec=get_float(node, '~ply_tf_tolerance_sec', 0.0, 'Tolerance window for exact-time PLY TF lookup. Zero requires an exact transform stamp.', min_value=0.0),
    )

    output_dir = config.ply_output_dir or getattr(node, "debug_output_dir", "")
    if output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        config.ply_output_dir = output_dir
    return config
