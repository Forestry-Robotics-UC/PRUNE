#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# ENTFAC Sensor Fusion implementation.
#
# Note:
#   This file was developed specifically for ENTFAC Sensor Fusion.
#   Project-level upstream attribution is documented in README.md.
#
# Modified by:
#   Duda Andrada (ENTFAC Sensor Fusion)
#
# Author: Duda Andrada
# Maintainer: Duda Andrada <duda.andrada@isr.uc.pt>
# License: GNU General Public License v3.0 (GPL-3.0)
# Repository: ENTFAC-Sensor-Fusion
#
# Description:
#   Convenience exports for small numpy helpers.

from entfac_fusion_core.utils.validation import (
    ensure_float_matrix,
    flatten_masked,
    require_homogeneous_transform,
)
from entfac_fusion_core.utils.semantics import (
    count_semantic_groups,
    count_unique_colors,
    count_unique_labels,
    unique_color_triplets,
    unique_label_ids,
)
from entfac_fusion_core.utils.masks import (
    apply_invalid_projection_samples,
    dilate_mask,
    invalid_image_to_mask,
    sample_invalid_mask,
)

__all__ = [
    "apply_invalid_projection_samples",
    "count_semantic_groups",
    "count_unique_colors",
    "count_unique_labels",
    "dilate_mask",
    "ensure_float_matrix",
    "flatten_masked",
    "invalid_image_to_mask",
    "require_homogeneous_transform",
    "sample_invalid_mask",
    "unique_color_triplets",
    "unique_label_ids",
]
