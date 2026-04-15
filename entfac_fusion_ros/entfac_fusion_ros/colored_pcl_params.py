"""Parameter helpers for the ENTFAC colored point-cloud ROS node."""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any, Optional, Tuple

import numpy as np
import rospy

from entfac_fusion_core.utils.validation import require_homogeneous_transform


def coerce_bool(val: Any) -> bool:
    """Normalize common ROS/YAML boolean representations."""
    if isinstance(val, bool):
        return val
    if isinstance(val, (int, float)):
        return bool(val)
    if isinstance(val, str):
        return val.strip().lower() in ("1", "true", "yes", "on")
    return bool(val)


def rosargv_has_private_param(name: str) -> bool:
    """Return ``True`` when ``name`` was provided as a private ROS CLI override."""
    if not isinstance(name, str):
        return False
    key = name
    if key.startswith("~"):
        key = key[1:]
    if "/" in key:
        key = key.rsplit("/", 1)[-1]
    prefix = f"_{key}:="
    return any(arg.startswith(prefix) for arg in sys.argv)


def record_param(node, name, value, source, description) -> None:
    """Record parameter provenance for the startup debug report."""
    node._param_meta[name] = {
        "value": value,
        "source": source,
        "description": description,
    }


def get_param(node, name, default, description, *, allow_empty=False):
    """Read a ROS parameter while tracking its origin for debug reporting."""
    has = rospy.has_param(name)
    if has:
        value = rospy.get_param(name)
        source = "param_server"
    else:
        value = default
        source = "default"
    if rosargv_has_private_param(name):
        source = "cli"
    if not allow_empty and value in ("", None):
        value = default
    record_param(node, name, value, source, description)
    return value


def get_param_str(node, name, default, description, *, allow_empty=False):
    """Read a string ROS parameter and store its normalized value."""
    raw = get_param(node, name, default, description, allow_empty=allow_empty)
    val = None if raw is None else str(raw)
    node._param_meta[name]["value"] = val
    return val


def get_param_bool(node, name, default, description):
    """Read a boolean ROS parameter and store its normalized value."""
    raw = get_param(node, name, default, description, allow_empty=True)
    val = coerce_bool(raw)
    node._param_meta[name]["value"] = val
    return val


def get_param_int(node, name, default, description):
    """Read an integer ROS parameter and store its normalized value."""
    raw = get_param(node, name, default, description)
    val = int(raw)
    node._param_meta[name]["value"] = val
    return val


def get_param_float(node, name, default, description):
    """Read a floating-point ROS parameter and store its normalized value."""
    raw = get_param(node, name, default, description)
    val = float(raw)
    node._param_meta[name]["value"] = val
    return val


def get_matrix_param(node, name, description):
    """Read a 4x4 homogeneous transform parameter from ROS."""
    has = rospy.has_param(name)
    raw = rospy.get_param(name, [])
    source = "param_server" if has else "default"
    mat = None
    if isinstance(raw, list) and len(raw) == 16:
        try:
            mat = require_homogeneous_transform(
                np.asarray(raw, dtype=float).reshape(4, 4)
            )
        except ValueError as exc:
            node._log.warn("_get_matrix_param", "%s rejected: %s", name, exc)
    elif raw not in (None, [], {}):
        node._log.warn(
            "_get_matrix_param",
            "%s expected 16-element list (row-major 4x4), got: %r",
            name,
            raw,
        )
    record_param(node, name, mat, source, description)
    return mat


def get_color_map(node, name, description):
    """Read an optional label->RGB dictionary from ROS."""
    has = rospy.has_param(name)
    raw = rospy.get_param(name, {})
    source = "param_server" if has else "default"
    color_map = None
    if isinstance(raw, dict):
        parsed = {}
        for key, value in raw.items():
            try:
                key_int = int(key)
                if isinstance(value, (list, tuple)) and len(value) == 3:
                    parsed[key_int] = [int(value[0]), int(value[1]), int(value[2])]
            except Exception:  # noqa: BLE001
                continue
        color_map = parsed if parsed else None
    record_param(node, name, color_map, source, description)
    return color_map


def load_camera_info_txt(
    node,
    txt_path: str,
) -> Tuple[np.ndarray, str, Optional[np.ndarray], str, Tuple[int, int], str]:
    """Load intrinsics and optional distortion coefficients from a text/YAML file."""
    path = Path(str(txt_path)).expanduser()
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()
    else:
        path = path.resolve()
    if not path.is_file():
        raise ValueError(f"~camera_info_txt file not found: {path}")

    text = path.read_text(encoding="utf-8", errors="ignore")
    number_pat = r"[-+]?(?:\d+\.\d*|\d*\.\d+|\d+)(?:[eE][-+]?\d+)?"

    def _extract_list(keys: Tuple[str, ...]) -> Optional[list]:
        for key in keys:
            pattern = rf"(?is)(?:^|[\r\n])\s*{re.escape(key)}\s*[:=]\s*\[([^\]]+)\]"
            match = re.search(pattern, text)
            if match:
                vals = [float(v) for v in re.findall(number_pat, match.group(1))]
                if vals:
                    return vals
        return None

    def _extract_scalar(keys: Tuple[str, ...]) -> str:
        for key in keys:
            pattern = (
                rf"(?im)^\s*{re.escape(key)}\s*[:=]\s*"
                r"['\"]?([^'\"\s#]+)['\"]?\s*(?:#.*)?$"
            )
            match = re.search(pattern, text)
            if match:
                return match.group(1).strip()
        return ""

    def _extract_float(keys: Tuple[str, ...]) -> Optional[float]:
        for key in keys:
            pattern = (
                rf"(?im)^\s*{re.escape(key)}\s*[:=]\s*"
                rf"({number_pat})\s*(?:#.*)?$"
            )
            match = re.search(pattern, text)
            if match:
                return float(match.group(1))
        return None

    def _extract_int(keys: Tuple[str, ...]) -> int:
        for key in keys:
            pattern = rf"(?im)^\s*{re.escape(key)}\s*[:=]\s*([-+]?\d+)\s*(?:#.*)?$"
            match = re.search(pattern, text)
            if match:
                return int(match.group(1))
        return 0

    k_vals = _extract_list(("K", "k", "intrinsics", "camera_matrix_data"))
    if k_vals is None:
        match = re.search(
            r"(?is)camera_matrix\s*:.*?data\s*:\s*\[([^\]]+)\]",
            text,
        )
        if match:
            k_vals = [float(v) for v in re.findall(number_pat, match.group(1))]
    if not k_vals:
        fx = _extract_float(("fx",))
        fy = _extract_float(("fy",))
        cx = _extract_float(("cx",))
        cy = _extract_float(("cy",))
        if None not in (fx, fy, cx, cy):
            k_vals = [
                float(fx),
                0.0,
                float(cx),
                0.0,
                float(fy),
                float(cy),
                0.0,
                0.0,
                1.0,
            ]
    if not k_vals:
        p_vals = _extract_list(("P", "projection_matrix_data"))
        if p_vals is None:
            match = re.search(
                r"(?is)projection_matrix\s*:.*?data\s*:\s*\[([^\]]+)\]",
                text,
            )
            if match:
                p_vals = [float(v) for v in re.findall(number_pat, match.group(1))]
        if p_vals and len(p_vals) >= 12:
            p = np.asarray(p_vals[:12], dtype=float).reshape(3, 4)
            k_vals = p[:, :3].reshape(-1).tolist()
    if not k_vals:
        all_numbers = [float(v) for v in re.findall(number_pat, text)]
        if len(all_numbers) == 9:
            k_vals = all_numbers
    if not k_vals or len(k_vals) < 9:
        raise ValueError(
            "~camera_info_txt does not contain intrinsics K. Expected "
            "K/camera_matrix.data list or fx/fy/cx/cy scalars."
        )

    intrinsics = np.asarray(k_vals[:9], dtype=float).reshape(3, 3)
    frame_id = _extract_scalar(("frame_id", "camera_frame", "camera_frame_id"))
    distortion_model = _extract_scalar(("distortion_model",)).lower()
    d_vals = _extract_list(("D", "distortion", "distortion_coefficients_data"))
    if d_vals is None:
        match = re.search(
            r"(?is)distortion_coefficients\s*:.*?data\s*:\s*\[([^\]]+)\]",
            text,
        )
        if match:
            d_vals = [float(v) for v in re.findall(number_pat, match.group(1))]
    distortion = np.asarray(d_vals, dtype=float).reshape(-1) if d_vals else None
    width = max(0, int(_extract_int(("image_width", "width"))))
    height = max(0, int(_extract_int(("image_height", "height"))))
    if width == 0 or height == 0:
        node._log.warn(
            "_load_camera_info_txt",
            "camera_info_txt=%s has no valid image width/height. Undistort may be disabled.",
            path,
        )
    return intrinsics, frame_id, distortion, distortion_model, (height, width), str(path)
