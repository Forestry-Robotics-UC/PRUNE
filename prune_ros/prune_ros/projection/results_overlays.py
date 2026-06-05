#!/usr/bin/env python3
"""ROS-agnostic helpers for results overlay export.

Call save_frame_overlays() from a post-processing tool that has access to the
raw image, projected UV coordinates, and rejection masks for each frame.
The node writes per-frame metrics CSVs; the overlay tool reads those and any
saved intermediate numpy data to render the PNGs independently.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Set, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Guard: check whether a frame has any projected samples worth exporting.
# ---------------------------------------------------------------------------

def has_overlay_projection_samples(
    *,
    u: np.ndarray,
    v: np.ndarray,
    z_m: Optional[np.ndarray],
) -> bool:
    """Return True when an overlay frame has projected samples worth exporting."""
    if z_m is None:
        return False
    u_arr = np.asarray(u).reshape(-1)
    v_arr = np.asarray(v).reshape(-1)
    z_arr = np.asarray(z_m).reshape(-1)
    return bool(u_arr.size > 0 and v_arr.size > 0 and z_arr.size > 0)


# ---------------------------------------------------------------------------
# Primitive drawing helpers.
# ---------------------------------------------------------------------------

def draw_projection_points(
    image: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    mask: np.ndarray,
    color: Tuple[int, int, int],
    dot_radius: int = 0,
    cv2=None,
) -> None:
    """Draw selected projected LiDAR points on *image* in-place (RGB)."""
    if u.size == 0:
        return
    h, w = image.shape[:2]
    selected = np.asarray(mask, dtype=bool).reshape(-1)
    if selected.shape[0] != u.shape[0]:
        return
    uu = u[selected]
    vv = v[selected]
    in_bounds = (uu >= 0) & (uu < w) & (vv >= 0) & (vv < h)
    uu = uu[in_bounds]
    vv = vv[in_bounds]
    if dot_radius <= 0:
        image[vv, uu] = np.asarray(color, dtype=np.uint8)
    elif cv2 is not None:
        rgb = (int(color[0]), int(color[1]), int(color[2]))
        for x, y in zip(uu.tolist(), vv.tolist()):
            cv2.circle(image, (x, y), dot_radius, rgb, thickness=-1, lineType=cv2.LINE_AA)


def blend_mask(
    image: np.ndarray,
    mask: np.ndarray,
    color: Tuple[int, int, int],
    alpha: float,
) -> np.ndarray:
    """Return a copy of *image* with pixels under *mask* blended toward *color*."""
    out = np.asarray(image, dtype=np.uint8).copy()
    mask = np.asarray(mask, dtype=bool)
    if mask.shape != out.shape[:2]:
        return out
    color_arr = np.asarray(color, dtype=np.float32)
    out_f = out.astype(np.float32, copy=False)
    out_f[mask] = (1.0 - alpha) * out_f[mask] + alpha * color_arr
    return np.clip(out_f, 0.0, 255.0).astype(np.uint8)


# ---------------------------------------------------------------------------
# Depth colour-map (near=red → orange → yellow-green → cyan → far=blue).
# Matches the palette in tools/results/make_gimp_layers.py (_DEPTH_LUT).
# ---------------------------------------------------------------------------

_DEPTH_WP = np.array([
    [0.90, 0.10, 0.05],
    [1.00, 0.60, 0.00],
    [0.75, 1.00, 0.00],
    [0.00, 0.85, 0.95],
    [0.05, 0.10, 0.90],
], dtype=np.float32)
_DEPTH_WP_T = np.array([0.0, 0.25, 0.5, 0.75, 1.0], dtype=np.float32)
_LUT_T = np.linspace(0.0, 1.0, 256, dtype=np.float32)
DEPTH_LUT = np.clip(
    np.stack([np.interp(_LUT_T, _DEPTH_WP_T, _DEPTH_WP[:, c]) for c in range(3)], axis=1) * 255,
    0, 255,
).astype(np.uint8)


# ---------------------------------------------------------------------------
# Full per-frame overlay writer.
# ---------------------------------------------------------------------------

def save_frame_overlays(
    *,
    out_dir: Path,
    frame_index: int,
    base_rgb: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    z_m: Optional[np.ndarray] = None,
    invalid_reject: np.ndarray,
    depth_edge_reject: np.ndarray,
    occlusion_reject: np.ndarray,
    keep: np.ndarray,
    depth_edge_map: Optional[np.ndarray] = None,
    invalid_mask: Optional[np.ndarray] = None,
    depth_edge_thresh: float = 0.05,
    save_types: Optional[Set[str]] = None,
    dot_radius: int = 2,
    cv2=None,
) -> None:
    """Render and save overlay PNGs for one frame.

    Parameters
    ----------
    out_dir:
        Directory to write PNGs into (created if absent).
    frame_index:
        Frame counter; used in the output filename.
    base_rgb:
        Clean undistorted camera image (H, W, 3) in RGB.
    u, v:
        Pixel coordinates of projected LiDAR points.
    z_m:
        Depth (metres) for each projected point; used for depth-colour overlay.
    invalid_reject, depth_edge_reject, occlusion_reject, keep:
        Boolean arrays aligned with u/v giving per-gate rejection and acceptance.
    depth_edge_map:
        Float edge-response map (H, W); blended into the depth-edge overlay.
    invalid_mask:
        Boolean mask (H, W); blended into the sky overlay.
    depth_edge_thresh:
        Threshold applied to *depth_edge_map* for the orange blend region.
    save_types:
        Subset of overlay names to write.  None → write all.
    dot_radius:
        Filled-circle radius in pixels (0 = single pixel).
    cv2:
        cv2 module; required for dot_radius > 0 and for writing PNGs.
    """
    if cv2 is None:
        import cv2 as _cv2
        cv2 = _cv2

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    base = np.ascontiguousarray(np.asarray(base_rgb, dtype=np.uint8).copy())
    if base.ndim != 3 or base.shape[2] != 3:
        return

    u = np.asarray(u, dtype=np.int32).reshape(-1)
    v = np.asarray(v, dtype=np.int32).reshape(-1)
    z_m_arr = None if z_m is None else np.asarray(z_m, dtype=np.float32).reshape(-1)

    if not has_overlay_projection_samples(u=u, v=v, z_m=z_m_arr):
        return

    invalid_reject = np.asarray(invalid_reject, dtype=bool).reshape(-1)
    depth_edge_reject = np.asarray(depth_edge_reject, dtype=bool).reshape(-1)
    occlusion_reject = np.asarray(occlusion_reject, dtype=bool).reshape(-1)
    truly_accepted = np.asarray(keep, dtype=bool).reshape(-1) & ~depth_edge_reject & ~occlusion_reject

    def _draw(img, mask, color):
        draw_projection_points(img, u, v, mask, color, dot_radius, cv2)

    # projected_overlay: all points coloured by gate status.
    projected_overlay = base.copy()
    _draw(projected_overlay, truly_accepted, (0, 255, 0))
    _draw(projected_overlay, occlusion_reject, (255, 0, 255))
    _draw(projected_overlay, depth_edge_reject, (255, 165, 0))
    _draw(projected_overlay, invalid_reject, (255, 0, 0))

    # rejection_overlay: all rejections coloured by gate.
    rejected_any = invalid_reject | depth_edge_reject | occlusion_reject
    rejection_overlay = base.copy()
    _draw(rejection_overlay, rejected_any, (255, 255, 0))
    _draw(rejection_overlay, occlusion_reject, (255, 0, 255))
    _draw(rejection_overlay, depth_edge_reject, (255, 165, 0))
    _draw(rejection_overlay, invalid_reject, (255, 0, 0))

    # sky_overlay: blended sky region + only sky-rejected dots.
    sky_overlay = base.copy()
    if invalid_mask is not None:
        sky_overlay = blend_mask(sky_overlay, np.asarray(invalid_mask, dtype=bool), (255, 0, 0), 0.35)
    _draw(sky_overlay, invalid_reject, (255, 0, 0))

    # invalid_mask_overlay: sky region blend only.
    invalid_overlay = base.copy()
    if invalid_mask is not None:
        invalid_overlay = blend_mask(invalid_overlay, np.asarray(invalid_mask, dtype=bool), (255, 0, 0), 0.45)

    # depth_edge_overlay: orange edge heatmap + edge-rejected dots.
    edge_overlay = base.copy()
    if depth_edge_map is not None:
        edge_mask = np.asarray(depth_edge_map, dtype=np.float32) >= float(depth_edge_thresh)
        edge_overlay = blend_mask(edge_overlay, edge_mask, (255, 165, 0), 0.55)
    _draw(edge_overlay, depth_edge_reject, (255, 0, 0))

    # occlusion_overlay: only occlusion-rejected dots.
    occlusion_overlay = base.copy()
    _draw(occlusion_overlay, occlusion_reject, (255, 0, 255))

    # depth_overlay: per-point distance colourmap.
    depth_overlay = base.copy()
    if z_m_arr is not None and z_m_arr.size > 0:
        z_finite = z_m_arr[np.isfinite(z_m_arr)]
        z_min = float(z_finite.min()) if z_finite.size else 0.0
        z_max = float(z_finite.max()) if z_finite.size else 1.0
        z_range = max(z_max - z_min, 0.01)
        z_norm = np.clip((z_m_arr - z_min) / z_range, 0.0, 1.0)
        idx = np.clip(np.round(z_norm * 255).astype(int), 0, 255)
        depth_colors_rgb = DEPTH_LUT[idx]
        h_img, w_img = depth_overlay.shape[:2]
        in_bounds = (u >= 0) & (u < w_img) & (v >= 0) & (v < h_img)
        for i in np.where(in_bounds)[0]:
            rgb = (int(depth_colors_rgb[i, 0]), int(depth_colors_rgb[i, 1]), int(depth_colors_rgb[i, 2]))
            if dot_radius <= 0:
                depth_overlay[int(v[i]), int(u[i])] = rgb
            else:
                cv2.circle(depth_overlay, (int(u[i]), int(v[i])), dot_radius, rgb, thickness=-1, lineType=cv2.LINE_AA)

    # accepted_overlay: only truly accepted points.
    accepted_overlay = base.copy()
    _draw(accepted_overlay, truly_accepted, (0, 255, 0))

    outputs = {
        "base": base.copy(),
        "depth_overlay": depth_overlay,
        "accepted_overlay": accepted_overlay,
        "projected_overlay": projected_overlay,
        "sky_overlay": sky_overlay,
        "occlusion_overlay": occlusion_overlay,
        "invalid_mask_overlay": invalid_overlay,
        "depth_edge_overlay": edge_overlay,
        "rejection_overlay": rejection_overlay,
    }

    for suffix, image in outputs.items():
        if save_types is not None and suffix not in save_types:
            continue
        out_path = out_dir / f"frame_{frame_index:06d}_{suffix}.png"
        cv2.imwrite(str(out_path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

    if z_m_arr is not None and z_m_arr.size > 0:
        np.savez_compressed(
            str(out_dir / f"frame_{frame_index:06d}_depths.npz"),
            u=np.asarray(u, dtype=np.int32),
            v=np.asarray(v, dtype=np.int32),
            z_m=z_m_arr,
        )

    profile = {
        "frame_index": frame_index,
        "n_total": int(u.size),
        "n_accepted": int(np.count_nonzero(truly_accepted)),
        "n_rejected_invalid_mask": int(np.count_nonzero(invalid_reject)),
        "n_rejected_depth_edge": int(np.count_nonzero(depth_edge_reject)),
        "n_rejected_occlusion": int(np.count_nonzero(occlusion_reject)),
        "n_rejected_any": int(np.count_nonzero(rejected_any)),
    }
    with open(str(out_dir / f"frame_{frame_index:06d}_profile.json"), "w") as _f:
        json.dump(profile, _f, indent=2)


def write_selected_frames_manifest(
    results_dir: Path,
    bag_name: str,
    selected_frames: list,
) -> None:
    """Write a JSON manifest listing which frame indices had overlays saved."""
    manifest_path = Path(results_dir) / "selected_frames.json"
    payload = {
        "bag_name": bag_name,
        "selected_frame_indices": sorted(set(int(i) for i in selected_frames)),
    }
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
