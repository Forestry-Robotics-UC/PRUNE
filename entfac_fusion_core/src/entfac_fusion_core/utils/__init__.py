#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Author: Duda Andrada
# Maintainer: Duda Andrada <duda.andrada@isr.uc.pt>
# License: MIT License (open source, free to modify and redistribute)
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

__all__ = [
    "count_semantic_groups",
    "count_unique_colors",
    "count_unique_labels",
    "ensure_float_matrix",
    "flatten_masked",
    "require_homogeneous_transform",
    "unique_color_triplets",
    "unique_label_ids",
]
