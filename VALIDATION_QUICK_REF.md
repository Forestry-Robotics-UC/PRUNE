# Quick Reference: CurtMini Validation

## One-Command Validation (Quick Test)

```bash
cd /home/forestsphere/work_utils/ENTFAC-Sensor-Fusion-dev
tools/validate_curtmini_workflow.sh --check-resources
```

**What it does:**
- 86 MB test bag, 60 second replay
- Checks sync offsets between camera and LiDAR
- Generates diagnostics to `/tmp/bag_validation_*/`
- Reports results with interpretation

**Expected time:** ~5-10 minutes

---

## Understanding Results

### Sync Stats (Best Case)
```
Mean offset: -0.0015s   ✓ OK (within ±5ms)
Stdev: 0.0082s          ✓ OK (<10ms)
Range: [-0.0234, 0.0198]s  ✓ OK (all within 30ms window)
```

### Sync Stats (Issue Case)
```
Mean offset: +0.0250s   ❌ PROBLEM (>5ms, needs offset tuning)
Stdev: 0.0350s          ❌ PROBLEM (>20ms, sync jitter high)
```

**Fix:** Adjust in `curt_mini.yaml`:
```yaml
semantic_time_offset_sec: 0.025  # Set to mean offset
```

---

## Diagnosing Sky Bleeding on Thin Branches

### Visual Check
Replay in RViz:
```bash
# In Docker container:
source devel/setup.bash
roslaunch entfac_fusion_ros dataset_fusion.launch \
    use_urdf:=true \
    urdf_path:=<path_to_urdf> \
    dataset_config:=config/curt_mini.yaml \
    rviz:=true
```

Look for: Sky color (blue) on vegetation areas (green/brown)

### Parameter Tuning (in order of impact)

| Parameter | Default | Try | Effect |
|-----------|---------|-----|--------|
| `projection_depth_edge_radius_px` | 2 | 1 | Reduce edge dilation, helps thin branches |
| `projection_confidence_min` | 0.45 | 0.65 | Stricter confidence, filter weak colors |
| `projection_invalid_mask_dilate_px` | — | 0 | No mask dilation, sharper boundaries |

### Step 1: Reduce Dilation
```yaml
# curt_mini.yaml
projection_depth_edge_radius_px: 1  # was 2
```
Re-run validation, check if sky bleeding reduced.

### Step 2: Increase Confidence
```yaml
projection_confidence_min: 0.65  # was 0.45
```
Re-run validation.

### Step 3: Enable Invalid Mask
If Perception publishes invalid mask:
```yaml
projection_invalid_mask_topic: "/perception/invalid_mask"
projection_invalid_mask_value: 0  # Sky class
projection_invalid_mask_dilate_px: 0
```

---

## Parameter Cheat Sheet

### Synchronization
- `sync_slop_sec`: Pairing window (default 0.03s = 30ms)
- `semantic_time_offset_sec`: Signed offset to apply to semantic timestamps

### Projection
- `projection_patch_size`: Local image sampling (1=single pixel)
- `projection_confidence_min`: Min RGB confidence (0-1)
- `projection_reject_depth_edges`: Enable edge rejection (true/false)
- `projection_depth_edge_radius_px`: Edge mask dilation radius

### Occlusion
- `projection_occlusion_epsilon_m`: Z-buffer margin (meters)
- `projection_occlusion_radius_px`: Local min-depth check radius

### Invalid Mask (Optional Bridge from Perception)
- `projection_invalid_mask_topic`: ROS topic with mask (if not set, disabled)
- `projection_invalid_mask_value`: Value to treat as invalid (default 255)
- `projection_invalid_mask_dilate_px`: Dilation of invalid mask

---

## Workflow States

```
[START] → [Check Resources] → [Sync Analysis] → [Bag Replay] → [Review Diagnostics] → [Pass/Fail]
                                                                                              ↓
                                                                         [Adjust Params] ← ←←
```

### Passing Criteria

✓ **Sync offset:** Mean within ±5ms, stdev <10ms  
✓ **Projection:** No NaN or out-of-bounds errors in logs  
✓ **Masking:** Sky bleeding absent or minimal  
✓ **Confidence:** Color confidence map stable  

### If Not Passing

1. **Sync issues:** Adjust `semantic_time_offset_sec`, re-validate
2. **Projection errors:** Check camera_info, frame transforms, verify calibration
3. **Masking artifacts:** Try parameter tuning steps above
4. **Persistent issues:** Review full validation logs in `/tmp/bag_validation_*/`

---

## Full Workflow (After Quick Validation Passes)

```bash
# 1. Validate with small bag (this document)
tools/validate_curtmini_workflow.sh --check-resources

# 2. Validate with slim bag (~28 GB, 15 min)
tools/validate_curtmini_workflow.sh \
    --bag /mnt/t7_shield/ICNF/icnf_ikalibr_order_003_010_slim.bag \
    --duration 300 \
    --check-resources

# 3. If passes, test UFOMap ingestion
# (Next stage - to be scheduled after this validation passes)

# 4. After UFOMap stable, enable online calibration
# (Only after metrics are stable - currently disabled)
```

---

## Troubleshooting

### Docker Image Build Fails
```bash
docker-compose build --no-cache sensor-fusion-ros
```

### Bag Replay Hangs
Check disk space:
```bash
df -h /tmp /mnt/t7_shield
```

### Node Crashes
Review logs:
```bash
cat /tmp/bag_validation_*/colored_pcl_node.log | tail -50
```

### GPU Out of Memory
If running Perception too (uses GPU):
```bash
docker-compose run --rm \
    -e NVIDIA_VISIBLE_DEVICES=0 \
    sensor-fusion-ros bash
```

---

## Files Reference

| File | Purpose |
|------|---------|
| `tools/validate_curtmini_workflow.sh` | Main entry point |
| `tools/validation_runner.sh` | Docker wrapper |
| `tools/validate_bag_workflow.py` | Analysis engine |
| `docs/VALIDATION_WORKFLOW.md` | Full documentation |
| `entfac_fusion_ros/config/curt_mini.yaml` | CurtMini parameters |

---

## Next: Check Resources Before Running

```bash
cd /home/forestsphere/work_utils/ENTFAC-Sensor-Fusion-dev
tools/validate_curtmini_workflow.sh --check-resources
```

**Ready?** Report back with:
- Resource check output
- Whether you want to start with quick test or slim bag
- Any known issues to specifically look for

