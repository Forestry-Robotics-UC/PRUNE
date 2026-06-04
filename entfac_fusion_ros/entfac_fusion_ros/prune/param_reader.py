"""Parameter reader facade for prune node setup."""

from __future__ import annotations

from dataclasses import fields
from typing import Any, Optional, Tuple

import numpy as np

from entfac_fusion_ros.prune.params import (
    get_color_map as _get_color_map_helper,
    get_matrix_param as _get_matrix_param_helper,
    get_param as _get_param_helper,
    get_param_bool as _get_param_bool_helper,
    get_param_float as _get_param_float_helper,
    get_param_int as _get_param_int_helper,
    get_param_str as _get_param_str_helper,
    load_camera_info_txt as _load_camera_info_txt_helper,
    record_param as _record_param_helper,
)


class ParamReader:
    def __init__(self, node: Any):
        self._node = node

    def record_param(self, name, value, source, description):
        return _record_param_helper(self._node, name, value, source, description)

    def get_param(self, name, default, description, *, allow_empty=False):
        return _get_param_helper(
            self._node, name, default, description, allow_empty=allow_empty
        )

    def get_param_str(self, name, default, description, *, allow_empty=False):
        return _get_param_str_helper(
            self._node, name, default, description, allow_empty=allow_empty
        )

    def get_param_bool(self, name, default, description):
        return _get_param_bool_helper(self._node, name, default, description)

    def get_param_int(self, name, default, description):
        return _get_param_int_helper(self._node, name, default, description)

    def get_param_float(self, name, default, description):
        return _get_param_float_helper(self._node, name, default, description)

    def get_matrix_param(self, name, description):
        return _get_matrix_param_helper(self._node, name, description)

    def get_color_map(self, name, description):
        return _get_color_map_helper(self._node, name, description)

    def apply_loaded_config(self, config: Any) -> None:
        for field in fields(config):
            setattr(self._node, field.name, getattr(config, field.name))

    def load_camera_info_txt(
        self, txt_path: str
    ) -> Tuple[np.ndarray, str, Optional[np.ndarray], str, Tuple[int, int], str]:
        return _load_camera_info_txt_helper(self._node, txt_path)
