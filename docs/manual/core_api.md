# Core API Reference

## Fusion functions

```{autofunction} prune_core.colored_pcl.fuse_depth_semantics
```

```{autofunction} prune_core.colored_pcl.fuse_lidar_semantics
```

## Dataclasses

```{autoclass} prune_core.types.observations.SemanticObservation
:members:
```

```{autoclass} prune_core.types.observations.DepthObservation
:members:
```

```{autoclass} prune_core.types.observations.PointObservation
:members:
```

```{autoclass} prune_core.types.observations.SemanticPointCloud
:members:
```

## Validation helpers

```{autofunction} prune_core.utils.validation.ensure_float_matrix
```

```{autofunction} prune_core.utils.validation.require_homogeneous_transform
```

```{autofunction} prune_core.utils.validation.flatten_masked
```

## Mask helpers

```{autofunction} prune_core.utils.masks.invalid_image_to_mask
```

```{autofunction} prune_core.utils.masks.sample_invalid_mask
```

```{autofunction} prune_core.utils.masks.apply_invalid_projection_samples
```
