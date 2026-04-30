# CurtMini Bag Validation Workflow

This guide walks through validating the forestsphere branch with CurtMini rosbag data, focusing on:
- **Sync offsets**: Camera-to-LiDAR timestamp alignment
- **Projection accuracy**: LiDAR points correctly mapped to image plane
- **Masking edge cases**: Detection of sky bleeding on thin branches and other artifacts

## Overview

The validation workflow consists of three stages:

### Stage 1: Sync Offset Analysis
Computes nearest-neighbor timestamp deltas between camera and LiDAR topics to verify synchronization health.

**Key metrics:**
- Mean offset (should be ~0 if sensors are time-synced)
- Stdev (should be <10ms for stable fusion)
- Range (identify outliers)

**Configuration:** `curt_mini.yaml` parameters:
- `sync_slop_sec: 0.03` (30ms pairing window)
- `semantic_time_offset_sec: 0.0` (adjust if mean offset is non-zero)

### Stage 2: Projection Validation
Replays the bag through colored_pcl_node with debug output enabled.

**Captures:**
- Projection overlay (LiDAR on camera image)
- Depth/edge/heatmap diagnostics
- Node logs with projection errors

**Configuration:** `curt_mini.yaml` projection parameters:
- `projection_patch_size: 1` (local image sampling window)
- `projection_confidence_min: 0.45` (RGB confidence threshold)
- `projection_reject_depth_edges: true` (enable edge rejection)
- `projection_depth_edge_radius_px: 2` (dilation of edge mask)
- `projection_occlusion_epsilon_m: 0.10` (z-buffer margin)

### Stage 3: Masking Artifact Detection
Analyzes saved diagnostics for:
- **Sky bleeding**: High-confidence sky labels extending into vegetation
- **Depth edge artifacts**: Over-dilation affecting thin branches
- **Confidence anomalies**: Unexpected patterns in color confidence

**Known issue:** Sky is bleeding on thin branches → likely caused by:
1. Sky semantic segmentation mask extending beyond actual sky boundary
2. Depth edge dilation too aggressive (radius_px=2)
3. Missing projection_invalid_mask filtering

## Quick Start

### 1. Validate with Small Test Bag (86 MB, ~2 minutes)

```bash
cd /home/forestsphere/work_utils/ENTFAC-Sensor-Fusion-dev

# Run full validation workflow
tools/validate_curtmini_workflow.sh \
    --bag /mnt/t7_shield/ICNF/ICNF_curt_localization_50hz.bag \
    --duration 60 \
    --check-resources

# Results saved to: /tmp/bag_validation_YYYYMMDD_HHMMSS/
```

### 2. Review Results

```bash
cd /tmp/bag_validation_YYYYMMDD_HHMMSS

# Inspect sync analysis
cat sync_stats.json | python3 -m json.tool

# Check for projection errors
grep -i "error\|warning\|fail" colored_pcl_node.log

# Review bag topics
cat bag_info.txt
```

### 3. Analyze Specific Issues

#### Check Sync Offset
If mean offset is not close to 0:
```bash
# Adjust in curt_mini.yaml:
semantic_time_offset_sec: <measured_offset>

# Re-run validation to verify fix
```

#### Check Projection Errors
If you see reprojection failures:
1. Verify camera calibration is loaded (check camera_info topic)
2. Check sync offset first
3. Verify frame transforms are published (check TF tree)
4. Review projection parameters

#### Check Sky Bleeding on Thin Branches
The issue manifests as:
- Sky semantic labels (class ID, often 0 or 255) appearing on vegetation
- Confidence scores misaligned between color and segmentation
- Depth edge dilation extending mask beyond thin structures

**Diagnostic steps:**
1. Check semantic segmentation model output (ONNX quality)
2. Review `projection_invalid_mask_topic` filtering parameters
3. Examine depth edge dilation (`projection_depth_edge_radius_px`)
4. Visualize in RViz with colored point cloud + semantic overlay

**Potential fixes:**
- Increase `projection_confidence_min` to require higher confidence
- Reduce `projection_depth_edge_radius_px` to avoid over-dilation
- Enable `projection_invalid_mask_topic` with sky-specific masking
- Fine-tune semantic segmentation model for edge cases

## Advanced Usage

### Run with Custom Config

```bash
tools/validate_curtmini_workflow.sh \
    --bag /path/to/bag.bag \
    --duration 120
```

Create alternate config in `entfac_fusion_ros/config/validation.yaml`:
```yaml
# Copy from curt_mini.yaml and adjust parameters
projection_depth_edge_radius_px: 1  # Reduce dilation
projection_confidence_min: 0.60      # Stricter confidence
projection_invalid_mask_dilate_px: 0  # Test without mask dilation
```

Run with custom config:
```bash
# Edit the validation workflow to pass --config validation.yaml
```

### Run with Slim Dataset Bag (28 GB, ~15 minutes)

```bash
tools/validate_curtmini_workflow.sh \
    --bag /mnt/t7_shield/ICNF/icnf_ikalibr_order_003_010_slim.bag \
    --duration 300 \
    --check-resources
```

### Manual ROS Workflow (For Debugging)

If the orchestrated workflow has issues, run manually in Docker:

```bash
cd /home/forestsphere/work_utils/ENTFAC-Sensor-Fusion-dev

# Start Docker container
docker-compose run --rm -v /tmp/validation:/output sensor-fusion-ros bash

# Inside container:
source devel/setup.bash

# Terminal 1: Start ROS
roscore

# Terminal 2: Start node with debug
rosrun entfac_fusion_ros colored_pcl_node.py _debug:=true

# Terminal 3: Replay bag
rosbag play --clock /mnt/t7_shield/ICNF/ICNF_curt_localization_50hz.bag

# Terminal 4 (optional): Record output
rosbag record -O /output/output.bag /ouster/rgb_colored

# Monitor in Terminal 5:
rviz -d <config>  # or rqt_image_view to view projection overlay
```

## Known Issues & Fixes

### Issue: Sky bleeding on thin branches

**Diagnosis:**
- Visual inspection shows sky color/labels on vegetation
- Confidence map shows mismatch between projected RGB and semantic labels

**Root causes:**
1. Semantic segmentation model oversegments sky boundary
2. Depth edge dilation (`projection_depth_edge_radius_px`) too large
3. Invalid mask not applied or not filtering sky class

**Fixes to try (in order):**
1. **Reduce dilation:**
   ```yaml
   projection_depth_edge_radius_px: 1  # from 2
   ```

2. **Enable invalid mask with sky filtering:**
   ```yaml
   projection_invalid_mask_topic: "/perception/invalid_mask"
   projection_invalid_mask_dilate_px: 0
   projection_invalid_mask_value: 0  # Assuming 0 = sky class
   ```

3. **Increase confidence threshold:**
   ```yaml
   projection_confidence_min: 0.65  # from 0.45
   ```

4. **Check segmentation model:**
   - Verify ONNX model file is correct
   - Test model output on sample images
   - Consider model retraining if systematic edge errors

### Issue: High sync offset variability (stdev > 20ms)

**Diagnosis:**
- `sync_stats.json` shows high stdev in timestamp deltas

**Root causes:**
1. Sensors not hardware synchronized
2. USB/PCIe timing jitter
3. System load affecting message timestamp accuracy

**Fixes:**
1. Verify PTP/RTC synchronization on hardware
2. Check system time with `ntpdate -q pool.ntp.org`
3. Increase `sync_slop_sec` to accommodate jitter (tradeoff: might pair wrong frames)
4. Adjust `semantic_time_offset_sec` to align mean offset to zero

### Issue: Projection failures (NaN, out-of-bounds points)

**Diagnosis:**
- colored_pcl_node.log shows reprojection errors
- Output point cloud has many uncolored points

**Root causes:**
1. Camera calibration not loaded or incorrect
2. Frame transforms not published
3. Camera FOV does not match LiDAR scan
4. Extreme sync offset

**Fixes:**
1. Verify camera info is published:
   ```bash
   rostopic echo /camera/color/camera_info | head -20
   ```

2. Check frame transforms:
   ```bash
   rosrun tf view_frames
   dot -Tpng frames.pdf > frames.png
   ```

3. Verify `target_frame` matches your setup
4. Check sync offset and adjust if needed

## Expected Outputs

After running the workflow, expect:

```
/tmp/bag_validation_YYYYMMDD_HHMMSS/
├── bag_info.txt                  # Bag topic summary
├── topics.txt                    # Full topic list
├── sync_stats.json               # Timestamp offset analysis
├── validation.log                # Validation script output
├── colored_pcl_node.log          # Node debug output
└── rosbag_play.log               # Bag replay log
```

### Interpreting sync_stats.json

```json
{
  "topic_a": "/camera/color/image_raw",
  "topic_b": "/ouster/points",
  "mean_delta_sec": -0.0015,           // Camera ~1.5ms ahead of LiDAR
  "stdev_delta_sec": 0.0082,           // Jitter ±8ms
  "median_delta_sec": -0.0012,
  "min_delta_sec": -0.0234,
  "max_delta_sec": 0.0198,
  "percentile_5": -0.0156,
  "percentile_95": 0.0124
}
```

**Interpretation:**
- ✓ Good: mean ~0, stdev <10ms → sensors are synchronized
- ⚠ Caution: stdev 10-20ms → acceptable but consider sync tuning
- ❌ Problem: mean far from 0 → adjust `semantic_time_offset_sec`
- ❌ Problem: stdev >30ms → hardware sync issue or `sync_slop_sec` too small

## Next Steps (After Validation)

1. **If all checks pass:**
   - Proceed to UFOMap ingestion with full dataset bag
   - Enable projection diagnostics in RViz
   - Monitor map quality and occlusion handling

2. **If sync offset needs tuning:**
   - Use measured mean offset to set `semantic_time_offset_sec`
   - Re-validate with adjusted config
   - Proceed only after offset is within ±5ms

3. **If sky bleeding persists:**
   - Try projection parameter adjustments from fixes above
   - Capture RViz visualizations showing the issue
   - Consider enabling invalid-mask bridge from Perception

4. **Before online calibration:**
   - Verify bag metrics are stable across multiple runs
   - Test with different environmental conditions if possible
   - Ensure disabled online correction (keep `online_correction_enable: false`)
   - Capture baseline metrics for later comparison

## References

- [Sensor Fusion README](../README.md#configuration)
- [Parameter Guide](../docs/manual/parameters.md)
- [Projection Diagnostics](../docs/manual/architecture.md#colored_pcl_node)
- [ROS Time Synchronization](http://wiki.ros.org/roscpp/Overview/Time)

