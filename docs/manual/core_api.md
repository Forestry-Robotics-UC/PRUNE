# Core API Reference

## Fusion functions

```{autofunction} entfac_fusion_core.colored_pcl.fuse_depth_semantics
```

```{autofunction} entfac_fusion_core.colored_pcl.fuse_lidar_semantics
```

## Dataclasses

```{autoclass} entfac_fusion_core.types.observations.SemanticObservation
:members:
```

```{autoclass} entfac_fusion_core.types.observations.DepthObservation
:members:
```

```{autoclass} entfac_fusion_core.types.observations.PointObservation
:members:
```

```{autoclass} entfac_fusion_core.types.observations.SemanticPointCloud
:members:
```

## Validation helpers

```{autofunction} entfac_fusion_core.utils.validation.ensure_float_matrix
```

```{autofunction} entfac_fusion_core.utils.validation.require_homogeneous_transform
```

```{autofunction} entfac_fusion_core.utils.validation.flatten_masked
```

## Mask helpers

```{autofunction} entfac_fusion_core.utils.masks.invalid_image_to_mask
```

```{autofunction} entfac_fusion_core.utils.masks.sample_invalid_mask
```

```{autofunction} entfac_fusion_core.utils.masks.apply_invalid_projection_samples
```
