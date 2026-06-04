"""Semantic color configuration for prune."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .config_helpers import get_int


@dataclass
class ColorConfig:
    colorize_labels: bool
    color_map: dict
    random_color_seed: int
    num_labels: int
    semantic_color_quantization_step: int


def load_color_config(node: Any) -> ColorConfig:
    return ColorConfig(
        colorize_labels=node._get_param_bool('~colorize_labels', False, "If true, publish an extra PointCloud2 field 'rgb' (label palette in 'labels' mode; passthrough colors in 'rgb' mode)."),
        color_map=node._get_color_map(
            '~color_map',
            'Optional dict {label_id: [r,g,b]} used to colorize labels when ~semantic_input_type=\'labels\'. YAML keys must be quoted (e.g. "0": [0,0,0]).',
        ),
        random_color_seed=node._get_param_int('~random_color_seed', 1, 'Seed for deterministic random label palette when ~colorize_labels is true and ~color_map is empty.'),
        num_labels=node._get_param_int(
            '~num_labels',
            0,
            "Optional number of label IDs (0=auto from first label image). Used only when ~semantic_input_type='labels' and ~colorize_labels is true with empty ~color_map.",
        ),
        semantic_color_quantization_step=get_int(node, '~semantic_color_quantization_step', 1, 'Quantize RGB/BGR semantic images to nearest multiple of this step before packing for the PointCloud2 rgb field (helps with JPEG artifacts).', min_value=1),
    )
