# Online Calibration Methodology (Classical, Edge-Oriented)

## Scope

This document describes the online LiDAR-camera misalignment functionality added to
`prune_node.py` using only classical methods (no neural networks). The goal is
continuous calibration health monitoring and conservative online correction under edge
compute constraints.

## Problem Statement

Given:
- camera intrinsics `K`,
- nominal extrinsic transform `T_cam_lidar`,
- LiDAR points `P_l`,
- semantic image `I_s` (labels or RGB),

estimate a small online correction `ΔR(roll, pitch, yaw)` and produce:
- corrected projection transform `T_corr = ΔR * T_cam_lidar`,
- calibration health `h ∈ [0,1]`,
- calibration uncertainty `u ∈ [0,1]`.

The implementation is intentionally **rotation-only** for online correction (KISS
constraint) and bounded by configurable clamps.

## Measurement Model

For each selected frame:
1. Build semantic edge map `E_s` from `I_s`.
2. Rasterize LiDAR depth map under current transform and derive depth edge map `E_d`.
3. Compute alignment score with normalized cosine correlation:

`s = <E_s, E_d> / (||E_s|| * ||E_d|| + eps)`.

This is a robust, scale-normalized edge-consistency proxy commonly used in
registration/correlation pipelines [R4], [R5].

## Observability Gate

Online updates are performed only when scene observability is sufficient. The node
computes:
- `r_fov`: ratio of in-FOV projected LiDAR points,
- `d_sem`: semantic edge density above threshold,
- `d_dep`: depth-edge density above threshold.

Each density is normalized by minimum expected density and clipped to `[0,1]`.
Observability is:

`o = (r_fov * d_sem_norm * d_dep_norm)^(1/3)`.

This is a practical observability proxy inspired by observability-aware calibration
literature [R2], [R3], grounded in classical observability principles [R1].

## Online Correction Update

Let `θ = [roll, pitch, yaw]^T` be the online correction state in radians.

Every `N` frames:
1. Evaluate baseline score `s(θ)`.
2. For each axis `i`, compute central finite-difference gradient:
   - `g_i = (s(θ + δe_i) - s(θ - δe_i)) / (2δ)`.
3. Estimate curvature:
   - `h_i = (s(θ + δe_i) - 2s(θ) + s(θ - δe_i)) / δ^2`.
4. Update:
   - if `h_i < 0`: `θ_i <- θ_i + η * g_i / (|h_i| + eps)`,
   - else: `θ_i <- θ_i + η * g_i`.
5. Clamp `θ_i` to `[-θ_max, θ_max]`.

This follows the classical iterative nonlinear least-squares spirit (Gauss-Newton /
Levenberg-Marquardt family) with conservative damping and bounds [R6], [R7].

## Health and Uncertainty Head

The ROS-agnostic module `OnlineCalibrationHealth` computes:

1. EMA score:
`s_ema(t) = (1-a) s_ema(t-1) + a s_t`.

2. Stability:
`σ_t = std({s_{t-k}} over sliding window)`.

3. Confidence:
`c_t = exp(-σ_t / σ_scale) * gate(o_t)`.

4. Health:
`h_t = sigmoid((s_ema(t) - c0) / c_scale) * c_t`.

5. Uncertainty:
`u_t = max(1 - c_t, u_corr_t)`.

`u_corr_t` comes from update curvature quality and increases when updates are gated.
This gives a conservative uncertainty output suited for fusion pipelines.

## Computational Design for Edge Deployment

The method is designed for predictable runtime:
- update every `~online_calibration_every_n_frames`,
- subsample to at most `~online_calibration_max_points`,
- avoid global bundle adjustment / heavy solvers,
- avoid neural inference.

This preserves online behavior while still reacting to gradual extrinsic drift.

## Integration in This Repository

Main integration points:
- Core health estimator:
  `entfac_fusion_core/calibration/online_health.py`
- ROS update loop and projection correction:
  `entfac_fusion_ros/prune_node.py`
- Tunable parameters:
  `entfac_fusion_ros/config/expert.yaml`

Debug topics:
- `/debug/calibration_health` (`Float32`)
- `/debug/calibration_uncertainty` (`Float32`)

## Assumptions and Limitations

- Small-angle drift assumption (bounded rotational correction).
- Translation drift is not corrected online in this stage.
- Edge-based score can degrade in low-texture scenes, rain/fog, or severe rolling-shutter artifacts.
- Observability gate intentionally prioritizes safety over aggressive correction.

## Suggested Evaluation Protocol (Paper-Ready)

1. Inject synthetic extrinsic angular errors into known-good sequences.
2. Measure recovery:
   - angular correction error over time,
   - convergence time,
   - final health/uncertainty calibration quality.
3. Report compute:
   - callback latency distribution,
   - update rate under edge hardware constraints.
4. Ablations:
   - no gate vs gate,
   - no curvature vs curvature-aware step,
   - different `N`, point budgets, and clamp values.

## References

[R1] Hermann, R., and Krener, A. J. "Nonlinear Controllability and Observability."
IEEE Transactions on Automatic Control, 1977. DOI: https://doi.org/10.1109/TAC.1977.1101601

[R2] Fu, B. et al. "LiDAR-Camera Calibration under Arbitrary Configurations:
Observability and Methods." arXiv:1903.06141 (2019). https://arxiv.org/abs/1903.06141

[R3] Lv, J. et al. "Observability-Aware Intrinsic and Extrinsic Calibration of
LiDAR-IMU Systems." arXiv:2205.03276 (2022). https://arxiv.org/abs/2205.03276

[R4] Lucas, B. D., and Kanade, T. "An Iterative Image Registration Technique with
an Application to Stereo Vision." IWU, 1981.
Curated reference page: https://idl.uw.edu/living-papers-paper/lucas-kanade/

[R5] Lewis, J. P. "Fast Normalized Cross-Correlation." Vision Interface, 1995.

[R6] Levenberg, K. "A Method for the Solution of Certain Non-Linear Problems in
Least Squares." Quarterly of Applied Mathematics, 1944.
DOI: https://doi.org/10.1090/qam/10666

[R7] Marquardt, D. W. "An Algorithm for Least-Squares Estimation of Nonlinear
Parameters." SIAM Journal on Applied Mathematics, 1963.
DOI: https://doi.org/10.1137/0111030

[R8] Xia, Z.-X. et al. "Robust Long-Range Perception Against Sensor Misalignment in
Autonomous Vehicles." WACV, 2025.
Open-access page:
https://openaccess.thecvf.com/content/WACV2025/html/Xia_Robust_Long-Range_Perception_Against_Sensor_Misalignment_in_Autonomous_Vehicles_WACV_2025_paper.html
