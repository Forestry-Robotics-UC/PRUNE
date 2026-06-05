"""Shared helpers for prune parameter loader modules."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional


def get_int(node: Any, name: str, default: int, description: str, *, min_value: Optional[int] = None, max_value: Optional[int] = None) -> int:
    value = node._get_param_int(name, default, description)
    if min_value is not None and value < min_value:
        raise ValueError(f"{name} must be >= {min_value}")
    if max_value is not None and value > max_value:
        raise ValueError(f"{name} must be <= {max_value}")
    return value


def get_float(node: Any, name: str, default: float, description: str, *, min_value: Optional[float] = None, max_value: Optional[float] = None) -> float:
    value = node._get_param_float(name, default, description)
    if min_value is not None and value < min_value:
        raise ValueError(f"{name} must be >= {min_value}")
    if max_value is not None and value > max_value:
        raise ValueError(f"{name} must be <= {max_value}")
    return value


def resolve_default_output_dir(node: Any) -> str:
    base_dir = Path.cwd() / 'output'
    node_name = getattr(node, '_node_name', 'prune_node')
    return str(base_dir / node_name)
