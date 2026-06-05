#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Author: Duda Andrada
# Maintainer: Duda Andrada <duda.andrada@isr.uc.pt>
# License: GNU General Public License v3.0 (GPL-3.0)
# Repository: PRUNE
#
# Description:
#   Unit tests for v1 public validation helpers (transforms, masks, matrices).

import sys
from pathlib import Path

import numpy as np
import pytest

CORE_SRC = Path(__file__).resolve().parents[1] / "prune_core" / "src"
if str(CORE_SRC) not in sys.path:
    sys.path.insert(0, str(CORE_SRC))

from prune_core.utils.validation import (  # noqa: E402
    ensure_float_matrix,
    flatten_masked,
    require_homogeneous_transform,
)


def test_ensure_float_matrix_rejects_shape_and_nonfinite():
    with pytest.raises(ValueError):
        ensure_float_matrix(np.zeros((2, 2)), (3, 3))
    with pytest.raises(ValueError):
        ensure_float_matrix(np.array([[np.nan]]), (1, 1))


def test_require_homogeneous_transform_rejects_invalid_last_row():
    tf = np.eye(4, dtype=float)
    tf[3, 3] = 0.0
    with pytest.raises(ValueError):
        require_homogeneous_transform(tf)


def test_require_homogeneous_transform_rejects_non_orthonormal_rotation():
    tf = np.eye(4, dtype=float)
    tf[:3, :3] = np.array(
        [[1.0, 0.1, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=float
    )
    with pytest.raises(ValueError):
        require_homogeneous_transform(tf)


def test_flatten_masked_requires_boolean_mask():
    values = np.arange(4).reshape(2, 2)
    with pytest.raises(ValueError):
        flatten_masked(values, np.ones((2, 2), dtype=np.uint8))
    mask = np.array([[True, False], [False, True]], dtype=bool)
    assert flatten_masked(values, mask).tolist() == [0, 3]

