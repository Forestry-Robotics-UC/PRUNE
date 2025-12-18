# Glossary

- **Measurement**: single-frame output from Sensor Fusion (no history).
- **Map / mapping update**: temporal accumulation performed by a separate mapping layer (out of scope here).
- **SemanticObservation**: per-pixel labels (+ optional confidence) aligned to the geometry input.
- **DepthObservation**: depth image aligned to semantic labels (meters in core fusion functions).
- **PointObservation**: unordered 3D points (e.g., LiDAR) in the sensor frame.
- **SemanticPointCloud**: output measurement containing `points_xyz` + `labels` (+ optional confidence).
- **Intrinsics (`K`)**: 3×3 camera calibration matrix.
- **Extrinsics / TF**: rigid transforms between sensor frames, represented as 4×4 homogeneous matrices.
- **`target_frame`**: output coordinate frame for the published `PointCloud2`.
- **Depth mode**: semantic labels + depth image → 3D points via back-projection.
- **LiDAR mode**: project LiDAR points into the image and sample semantic pixels.
- **`semantic_input_type`**
  - `labels`: semantic image encodes integer class IDs.
  - `rgb`: semantic image encodes colors; node can pass colors through for visualization (label IDs are not available).

