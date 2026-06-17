#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Author: Duda Andrada
# Maintainer: Duda Andrada <duda.andrada@isr.uc.pt>
# License: GNU General Public License v3.0 (GPL-3.0)
# Repository: PRUNE
#
# Description:
#   Offline parameter extractor for prune_node (used for documentation).

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


PACKAGE_DIR = (
    Path(__file__).resolve().parents[2]
    / "prune_ros"
    / "prune_ros"
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
    getter_types = {
        "_get_param_str": "str",
        "_get_param_bool": "bool",
        "_get_param_int": "int",
        "_get_param_float": "float",
        "_get_param": "raw",
        "_get_color_map": "dict",
        "_get_matrix_param": "list[16]",
        "get_float": "float",
        "get_int": "int",
    }

    rows_by_name: Dict[str, Tuple[str, str, str, str]] = {}
    for path in sorted(PACKAGE_DIR.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            attr = func.attr if isinstance(func, ast.Attribute) else (
                func.id if isinstance(func, ast.Name) else None
            )
            if attr not in getter_types:
                continue
            if not node.args:
                continue
            # get_float(node, name, default, desc, ...) vs self._get_param_x(name, default, desc)
            arg_offset = 1 if attr in ("get_float", "get_int") else 0
            if len(node.args) <= arg_offset:
                continue
            name = _const(node.args[arg_offset])
            if not isinstance(name, str) or not name.startswith("~"):
                continue

            if attr in ("_get_color_map", "_get_matrix_param"):
                default = None
                desc = _const(node.args[arg_offset + 1]) if len(node.args) > arg_offset + 1 else ""
            else:
                default = _const(node.args[arg_offset + 1]) if len(node.args) > arg_offset + 1 else None
                desc = _const(node.args[arg_offset + 2]) if len(node.args) > arg_offset + 2 else ""
            if not isinstance(desc, str):
                desc = ""
            rows_by_name[name] = (name, getter_types[attr], _format_default(default), desc)

    # Stable order for docs.
    return sorted(rows_by_name.values(), key=lambda r: r[0])


def main() -> None:
    rows = _extract()
    print("| Param | Type | Default | Description |")
    print("|---|---|---|---|")
    for name, typ, default, desc in rows:
        desc = desc.replace("\n", " ").strip()
        print(f"| `{name}` | `{typ}` | `{default}` | {desc} |")


if __name__ == "__main__":
    main()
