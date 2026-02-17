#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Derived from Semantic SLAM
#
# Original Author:
#   Xuan Zhang
#
# Subsequent Contributions:
#   David Russell
#
# Modified by:
#   Duda Andrada (ENTFAC Sensor Fusion)
#
# Original project:
#   https://github.com/floatlazer/semantic_slam
#
# Author: Duda Andrada
# Maintainer: Duda Andrada <duda.andrada@isr.uc.pt>
# License: GNU General Public License v3.0 (GPL-3.0)
# Repository: ENTFAC-Sensor-Fusion
#
# Description:
#   Offline parameter extractor for colored_pcl_node (used for documentation).

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


NODE_PATH = (
    Path(__file__).resolve().parents[2]
    / "entfac_fusion_ros"
    / "entfac_fusion_ros"
    / "colored_pcl_node.py"
)


def _const(node: ast.AST) -> Optional[Any]:
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, (ast.List, ast.Tuple)):
        vals = []
        for elt in node.elts:
            v = _const(elt)
            if v is None:
                return None
            vals.append(v)
        return vals
    if isinstance(node, ast.Dict):
        out: Dict[Any, Any] = {}
        for k, v in zip(node.keys, node.values):
            kk = _const(k)
            vv = _const(v)
            if kk is None or vv is None:
                return None
            out[kk] = vv
        return out
    return None


def _format_default(val: Any) -> str:
    if val is None:
        return "None"
    if isinstance(val, str):
        return repr(val)
    if isinstance(val, bool):
        return "true" if val else "false"
    return repr(val)


def _extract() -> List[Tuple[str, str, str, str]]:
    tree = ast.parse(NODE_PATH.read_text(encoding="utf-8"))
    rows: List[Tuple[str, str, str, str]] = []
    getter_types = {
        "_get_param_str": "str",
        "_get_param_bool": "bool",
        "_get_param_int": "int",
        "_get_param_float": "float",
        "_get_param": "raw",
        "_get_color_map": "dict",
        "_get_matrix_param": "list[16]",
    }

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not isinstance(func, ast.Attribute):
            continue
        if func.attr not in getter_types:
            continue
        if not node.args:
            continue
        name = _const(node.args[0])
        if not isinstance(name, str) or not name.startswith("~"):
            continue

        # Most getters use (name, default, description). Some use (name, description).
        if func.attr in ("_get_color_map", "_get_matrix_param"):
            default = None
            desc = _const(node.args[1]) if len(node.args) > 1 else ""
        else:
            default = _const(node.args[1]) if len(node.args) > 1 else None
            desc = _const(node.args[2]) if len(node.args) > 2 else ""
        if not isinstance(desc, str):
            desc = ""
        rows.append((name, getter_types[func.attr], _format_default(default), desc))

    # Stable order for docs.
    rows.sort(key=lambda r: r[0])
    return rows


def main() -> None:
    rows = _extract()
    print("| Param | Type | Default | Description |")
    print("|---|---|---|---|")
    for name, typ, default, desc in rows:
        desc = desc.replace("\n", " ").strip()
        print(f"| `{name}` | `{typ}` | `{default}` | {desc} |")


if __name__ == "__main__":
    main()
