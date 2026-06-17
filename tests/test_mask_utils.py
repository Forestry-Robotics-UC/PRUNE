#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Author: Duda Andrada
# Maintainer: Duda Andrada <duda.andrada@isr.uc.pt>
# License: GNU General Public License v3.0 (GPL-3.0)
# Repository: PRUNE
#
# Description:
#   Unit tests for projected invalid-mask helpers.

import sys
from pathlib import Path

import numpy as np
import pytest

CORE_SRC = Path(__file__).resolve().parents[1] / "prune_core" / "src"
if str(CORE_SRC) not in sys.path:
    sys.path.insert(0, str(CORE_SRC))

from prune_core.utils.masks import (  # noqa: E402
    apply_invalid_projection_samples,
    invalid_image_to_mask,
    sample_invalid_mask,
)


def test_invalid_image_to_mask_uses_configured_value_and_dilation():
    image = np.zeros((5, 5), dtype=np.uint8)
    image[2, 2] = 255
    image[4, 4] = 1

    mask = invalid_image_to_mask(image, invalid_value=255, dilate_px=1)

    assert mask.dtype == np.bool_
    assert mask[2, 2]
    assert mask[1, 2]
    assert mask[2, 1]
    assert not mask[0, 0]
    assert not mask[4, 4]


def test_invalid_image_to_mask_rejects_non_image_inputs():
    with pytest.raises(ValueError):
        invalid_image_to_mask(np.zeros((2, 2, 3), dtype=np.uint8))

    with pytest.raises(ValueError):
        invalid_image_to_mask(np.zeros((2, 2), dtype=np.float32), dilate_px=-1)


def test_invalid_image_to_mask_does_not_wrap_invalid_value_to_image_dtype():
    image = np.array([[255]], dtype=np.uint8)

    mask = invalid_image_to_mask(image, invalid_value=65535)

    assert not mask[0, 0]


def test_sample_invalid_mask_marks_invalid_and_out_of_bounds_pixels():
    mask = np.array(
        [
            [False, True],
            [False, False],
        ],
        dtype=bool,
    )

    invalid = sample_invalid_mask(
        mask,
        u=np.array([0, 1, -1, 0, 2]),
        v=np.array([0, 0, 0, 2, 0]),
    )

    assert invalid.tolist() == [False, True, True, True, True]


def test_apply_invalid_projection_samples_marks_semantics_unknown():
    labels = np.array([1, 2, 3], dtype=np.int64)
    confidence = np.array([0.8, 0.7, 0.6], dtype=np.float32)
    rgb_values = np.array([10.0, 20.0, 30.0], dtype=np.float32)
    invalid = np.array([False, True, False], dtype=bool)

    labels_out, confidence_out, rgb_out = apply_invalid_projection_samples(
        invalid,
        labels=labels,
        confidence=confidence,
        rgb_values=rgb_values,
    )

    assert labels_out.tolist() == [1, -1, 3]
    np.testing.assert_allclose(confidence_out, [0.8, 0.0, 0.6])
    np.testing.assert_allclose(rgb_out, [10.0, 0.0, 30.0])
    assert labels.tolist() == [1, 2, 3]
