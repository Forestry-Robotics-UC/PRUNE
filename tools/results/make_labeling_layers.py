#!/usr/bin/env python3
"""Produce manual-labeling layer sets from overlay PNGs + depth sidecars.

For each frame that has a *_depths.npz sidecar (written by prune_node
when save_results is enabled) this tool writes three files into a
labeling_layers/ sibling directory:

  frame_XXXXXX_base.png            Clean camera RGB — annotation dots removed.
  frame_XXXXXX_lidar_black.png     RGBA — all projected LiDAR points as solid
                                   black dots on a transparent background.
  frame_XXXXXX_lidar_depth.png     RGBA — same points coloured by distance
                                   using the depth palette (near=red, far=blue), transparent bg.
  frame_XXXXXX_depth_colorbar.png  Horizontal legend: distance range in metres.

The three PNGs are intended to be opened as GIMP layers (or any editor that
supports alpha compositing) so a human annotator can toggle the LiDAR overlays
on/off while drawing ground-truth labels on the clean base image.

Usage
-----
  python3 tools/results/make_labeling_layers.py --overlays-dir <path>
  python3 tools/results/make_labeling_layers.py --results-dir <path>
  python3 tools/results/make_labeling_layers.py --overlays-dir <path> --dot-radius 5
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Depth colormap: warm (near) → cool (far).  red→orange→yellow-green→cyan→blue
# Distinct at both ends; no dark-purple-on-both-ends ambiguity of Turbo.
# ---------------------------------------------------------------------------
_DEPTH_WAYPOINTS = np.array([
    [0.90, 0.10, 0.05],   # 0.00 — near: red
    [1.00, 0.60, 0.00],   # 0.25 — orange
    [0.75, 1.00, 0.00],   # 0.50 — yellow-green
    [0.00, 0.85, 0.95],   # 0.75 — cyan
    [0.05, 0.10, 0.90],   # 1.00 — far: blue
], dtype=np.float32)
_wp_t = np.array([0.0, 0.25, 0.5, 0.75, 1.0], dtype=np.float32)
_lut_t = np.linspace(0.0, 1.0, 256, dtype=np.float32)
_DEPTH_LUT: np.ndarray = np.clip(
    np.stack([
        np.interp(_lut_t, _wp_t, _DEPTH_WAYPOINTS[:, c])
        for c in range(3)
    ], axis=1) * 255,
    0, 255,
).astype(np.uint8)

# Known annotation colours the node paints (PIL RGB after BGR save):
_ANNOTATION_COLORS: list[tuple[int, int, int]] = [
    (0,   255, 0),
    (255, 0,   255),
    (0,   165, 255),
    (0,   0,   255),
    (0,   255, 255),
]


def _depth_rgb(t: np.ndarray) -> np.ndarray:
    """Map normalised [0,1] depth values to uint8 RGB. Near=red, far=blue."""
    return _DEPTH_LUT[np.clip(np.round(t * 255).astype(int), 0, 255)]


def _turbo_rgb(t: np.ndarray) -> np.ndarray:
    """Backward-compatible alias for the shared depth palette."""
    return _depth_rgb(t)


def _annotation_mask(rgb: np.ndarray) -> np.ndarray:
    mask = np.zeros(rgb.shape[:2], dtype=bool)
    for r, g, b in _ANNOTATION_COLORS:
        mask |= (rgb[:, :, 0] == r) & (rgb[:, :, 1] == g) & (rgb[:, :, 2] == b)
    return mask


def _paint_dots(canvas_rgba: np.ndarray, u: np.ndarray, v: np.ndarray,
                colors: np.ndarray, dot_r: int) -> None:
    """Paint filled circles of radius dot_r into canvas_rgba in-place."""
    h, w = canvas_rgba.shape[:2]
    offsets = [
        (dy, dx)
        for dy in range(-dot_r, dot_r + 1)
        for dx in range(-dot_r, dot_r + 1)
        if dy * dy + dx * dx <= dot_r * dot_r
    ]
    for dy, dx in offsets:
        vv = np.clip(v + dy, 0, h - 1)
        uu = np.clip(u + dx, 0, w - 1)
        canvas_rgba[vv, uu, :3] = colors
        canvas_rgba[vv, uu, 3] = 255


def _make_colorbar(z_min: float, z_max: float, dot_r: int, width: int = 512) -> Image.Image:
    gh, margin, tick_h, label_h = 40, 0, 6, 18
    total_h = gh + tick_h + label_h + 2
    img = Image.new("RGBA", (width, total_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    t = np.linspace(0.0, 1.0, width, dtype=np.float32)
    strip = Image.fromarray(_turbo_rgb(t)[np.newaxis, :, :].repeat(gh, axis=0), "RGB")
    img.paste(strip, (0, 0))
    draw.rectangle([0, 0, width - 1, gh - 1], outline=(200, 200, 200, 255))

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
    except OSError:
        font = ImageFont.load_default()

    for i in range(6):
        frac = i / 5
        x = int(frac * (width - 1))
        dist = z_min + frac * (z_max - z_min)
        label = f"{dist:.1f} m"
        draw.line([(x, gh), (x, gh + tick_h)], fill=(200, 200, 200, 255))
        bbox = draw.textbbox((0, 0), label, font=font)
        lw = bbox[2] - bbox[0]
        draw.text((max(0, min(x - lw // 2, width - lw)), gh + tick_h + 1),
                  label, fill=(220, 220, 220, 255), font=font)

    title = f"LiDAR depth · dot r={dot_r}px · near=red  far=blue"
    bbox = draw.textbbox((0, 0), title, font=font)
    draw.text((width // 2 - (bbox[2] - bbox[0]) // 2, gh + tick_h + 1),
              title, fill=(160, 160, 160, 200), font=font)

    return img


def process_frame(npz: Path, overlays_dir: Path, out_dir: Path,
                  *, dot_r: int, overwrite: bool) -> bool:
    """Process one frame. Returns True if any file was written."""
    frame_tag = npz.stem[: -len("_depths")]

    base_out = out_dir / f"{frame_tag}_base.png"
    black_out = out_dir / f"{frame_tag}_lidar_black.png"
    depth_out = out_dir / f"{frame_tag}_lidar_depth.png"

    all_exist = base_out.exists() and black_out.exists() and depth_out.exists()
    if not overwrite and all_exist:
        return False

    # Load depth sidecar
    data = np.load(npz)
    u: np.ndarray = data["u"].astype(np.int32)
    v: np.ndarray = data["v"].astype(np.int32)
    z_m: np.ndarray = data["z_m"].astype(np.float32)
    if z_m.size == 0:
        return False

    # Infer image dimensions from the projected_overlay PNG for this frame
    src_png = overlays_dir / f"{frame_tag}_projected_overlay.png"
    if not src_png.exists():
        # Fall back to any overlay PNG for this frame
        candidates = sorted(overlays_dir.glob(f"{frame_tag}_*.png"))
        if not candidates:
            return False
        src_png = candidates[0]

    rgb = np.asarray(Image.open(src_png).convert("RGB"), dtype=np.uint8)
    h, w = rgb.shape[:2]

    out_dir.mkdir(parents=True, exist_ok=True)

    # Base: strip annotation pixels
    if overwrite or not base_out.exists():
        base = rgb.copy()
        base[_annotation_mask(rgb)] = 0
        Image.fromarray(base, "RGB").save(base_out)

    # Black layer
    if overwrite or not black_out.exists():
        rgba = np.zeros((h, w, 4), dtype=np.uint8)
        black = np.zeros((len(u), 3), dtype=np.uint8)  # RGB = (0,0,0)
        _paint_dots(rgba, u, v, black, dot_r)
        Image.fromarray(rgba, "RGBA").save(black_out)

    # Depth layer + colorbar
    if overwrite or not depth_out.exists():
        z_lo = float(np.percentile(z_m, 2))
        z_hi = float(np.percentile(z_m, 98))
        if z_hi <= z_lo:
            z_hi = z_lo + 1.0
        t = np.clip((z_m - z_lo) / (z_hi - z_lo), 0.0, 1.0)
        colors = _turbo_rgb(t)

        rgba = np.zeros((h, w, 4), dtype=np.uint8)
        _paint_dots(rgba, u, v, colors, dot_r)
        Image.fromarray(rgba, "RGBA").save(depth_out)

        cb_path = out_dir / f"{frame_tag}_depth_colorbar.png"
        if overwrite or not cb_path.exists():
            _make_colorbar(z_lo, z_hi, dot_r, width=min(w, 512)).save(cb_path)

    return True


def process_overlays_dir(overlays_dir: Path, *, dot_r: int, overwrite: bool) -> int:
    npz_files = sorted(overlays_dir.glob("*_depths.npz"))
    if not npz_files:
        return 0
    out_dir = overlays_dir.parent / "labeling_layers"
    it = tqdm(npz_files, desc=str(overlays_dir.parent.name), unit="frame") if tqdm else npz_files
    written = sum(
        1 for npz in it
        if process_frame(npz, overlays_dir, out_dir, dot_r=dot_r, overwrite=overwrite)
    )
    return written


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--overlays-dir", type=Path)
    group.add_argument("--results-dir", type=Path,
                       help="Walk all .../overlays/ dirs under this path.")
    parser.add_argument("--dot-radius", type=int, default=4,
                        help="Dot radius in pixels for LiDAR point visualisation (default: 4).")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    dirs = (
        [args.overlays_dir]
        if args.overlays_dir
        else sorted(args.results_dir.rglob("overlays"))
    )

    total = 0
    for d in dirs:
        if not d.is_dir():
            continue
        n = process_overlays_dir(d, dot_r=args.dot_radius, overwrite=args.overwrite)
        if n:
            print(f"  {d.parent.name}/overlays → labeling_layers/ ({n} frames)")
        total += n

    print(f"\nDone. {total} frames processed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
