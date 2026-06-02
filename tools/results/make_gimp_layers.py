#!/usr/bin/env python3
"""Produce GIMP-ready layer pairs from composited overlay PNGs.

For each projected_overlay / rejection_overlay frame this tool writes two
files into a gimp_layers/ sibling directory:

  frame_XXXXXX_base.png              Clean camera RGB — annotation pixels zeroed.
  frame_XXXXXX_<suffix>_layer.png    RGBA — annotation pixels on transparent bg.

For blended-mask overlays (invalid_mask_overlay, depth_edge_overlay) only the
RGBA layer is written (the blend cannot be reversed without the original mask).

If frame_XXXXXX_depths.npz is present (written by colored_pcl_node when
save_results is enabled) two additional files are produced:

  frame_XXXXXX_depth_layer.png       RGBA — projected points coloured by
                                     distance using the depth palette, on a
                                     fully-transparent background.
  frame_XXXXXX_depth_colorbar.png    Horizontal gradient strip with distance
                                     tick labels in metres.

Annotation colour mapping
-------------------------
The node draws in RGB then saves with cv2.COLOR_RGB2BGR, so on-disk PIL reads:
  green   (0,255,0)    → accepted points
  magenta (255,0,255)  → occlusion-rejected
  sky-blue (0,165,255) → depth-edge-rejected  [orange in node RGB, swapped]
  blue    (0,0,255)    → invalid-mask-rejected [red in node RGB, swapped]
  cyan    (0,255,255)  → any-rejected          [yellow in node RGB, swapped]

Usage
-----
  python3 tools/results/make_gimp_layers.py --overlays-dir <path>
  python3 tools/results/make_gimp_layers.py --results-dir <path>   # all variants
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


# On-disk PIL-RGB values after the node's cv2.COLOR_RGB2BGR save:
_ANNOTATION_COLORS: list[tuple[int, int, int]] = [
    (0,   255, 0),    # accepted          (green)
    (255, 0,   255),  # occlusion-reject  (magenta)
    (0,   165, 255),  # depth-edge-reject (sky-blue; orange BGR-swapped)
    (0,   0,   255),  # invalid-reject    (blue;     red BGR-swapped)
    (0,   255, 255),  # any-rejected      (cyan;     yellow BGR-swapped)
]

# Per-gate layers written from projected_overlay: (label, on-disk PIL color)
_GATE_LAYERS: list[tuple[str, tuple[int, int, int]]] = [
    ("accepted",    (0,   255, 0)),    # green
    ("sky_mask",    (0,   0,   255)),  # blue  (node red, BGR-swapped)
    ("depth_edge",  (0,   165, 255)),  # sky-blue (node orange, BGR-swapped)
    ("occlusion",   (255, 0,   255)),  # magenta
]

_POINT_SUFFIXES = {"projected_overlay", "rejection_overlay"}
_BLEND_SUFFIXES = {"invalid_mask_overlay", "depth_edge_overlay"}


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

def _depth_rgb(t: np.ndarray) -> np.ndarray:
    """Map normalised [0,1] depth to uint8 RGB. Near=red, far=blue."""
    idx = np.clip(np.round(t * 255).astype(int), 0, 255)
    return _DEPTH_LUT[idx]


def _turbo_rgb(t: np.ndarray) -> np.ndarray:
    """Backward-compatible alias for the shared depth palette."""
    return _depth_rgb(t)


# ---------------------------------------------------------------------------
# Annotation helpers
# ---------------------------------------------------------------------------

def _annotation_mask(rgb: np.ndarray) -> np.ndarray:
    """Bool (H,W) — True where a pixel exactly matches any annotation colour."""
    mask = np.zeros(rgb.shape[:2], dtype=bool)
    for r, g, b in _ANNOTATION_COLORS:
        mask |= (rgb[:, :, 0] == r) & (rgb[:, :, 1] == g) & (rgb[:, :, 2] == b)
    return mask


def _process_point_overlay(rgb: np.ndarray, out_dir: Path, frame_tag: str, suffix: str, src_dir: Optional[Path] = None, overwrite: bool = False) -> None:
    """Write base.png (RGB, dots zeroed) and per-gate RGBA layers.

    For projected_overlay: writes one RGBA layer per gate (accepted, sky_mask,
    depth_edge, occlusion) — each layer has only that gate's dots on a fully
    transparent background.  In GIMP: stack them over the base layer.

    For other point overlays: writes a single combined RGBA layer as before.

    The base is shared across all point-overlay types for the same frame — only
    written once (projected_overlay is processed first alphabetically).
    """
    ann = _annotation_mask(rgb)

    base_path = out_dir / f"{frame_tag}_base.png"
    if not base_path.exists() or overwrite:
        # Prefer the clean base saved by the node (no dot residue by construction).
        src_base = (src_dir / f"{frame_tag}_base.png") if src_dir is not None else None
        if src_base is not None and src_base.exists():
            import shutil
            shutil.copy2(src_base, base_path)
        else:
            base = rgb.copy()
            base[ann] = 0  # fallback: zero annotation pixels
            Image.fromarray(base, "RGB").save(base_path)

    if suffix == "projected_overlay":
        # One RGBA layer per gate — transparent everywhere except that gate's dots.
        for gate_label, (r, g, b) in _GATE_LAYERS:
            gate_mask = (rgb[:, :, 0] == r) & (rgb[:, :, 1] == g) & (rgb[:, :, 2] == b)
            if not gate_mask.any():
                continue
            rgba = np.zeros((*rgb.shape[:2], 4), dtype=np.uint8)
            rgba[gate_mask, :3] = (r, g, b)
            rgba[gate_mask, 3] = 255
            Image.fromarray(rgba, "RGBA").save(
                out_dir / f"{frame_tag}_{gate_label}_layer.png"
            )
    else:
        rgba = np.zeros((*rgb.shape[:2], 4), dtype=np.uint8)
        rgba[ann, :3] = rgb[ann]
        rgba[ann, 3] = 255
        Image.fromarray(rgba, "RGBA").save(out_dir / f"{frame_tag}_{suffix}_layer.png")


def _process_blend_overlay(rgb: np.ndarray, out_dir: Path, frame_tag: str, suffix: str) -> None:
    """Write RGBA layer for blend-type overlays (opaque everywhere — full composite)."""
    rgba = np.dstack([rgb, np.full(rgb.shape[:2], 255, dtype=np.uint8)])
    Image.fromarray(rgba, "RGBA").save(out_dir / f"{frame_tag}_{suffix}_layer.png")


# ---------------------------------------------------------------------------
# Depth layer
# ---------------------------------------------------------------------------

_COLORBAR_W = 512
_COLORBAR_H = 40
_COLORBAR_MARGIN = 8    # px above gradient strip for tick labels
_COLORBAR_TICK_H = 6    # px tick line height below the gradient
_COLORBAR_TOTAL_H = _COLORBAR_H + _COLORBAR_MARGIN + 20  # room for labels


def _make_colorbar(z_min: float, z_max: float, dot_r: int) -> Image.Image:
    """Horizontal depth colorbar with distance labels in metres."""
    w, gh = _COLORBAR_W, _COLORBAR_H
    total_h = gh + _COLORBAR_MARGIN + 20

    img = Image.new("RGBA", (w, total_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    # Gradient strip
    t = np.linspace(0.0, 1.0, w, dtype=np.float32)
    rgb_strip = _turbo_rgb(t)  # (w, 3)
    strip = Image.fromarray(rgb_strip[np.newaxis, :, :].repeat(_COLORBAR_MARGIN, axis=0), "RGB")
    img.paste(strip.resize((w, gh)), (0, 0))

    # Border
    draw.rectangle([0, 0, w - 1, gh - 1], outline=(200, 200, 200, 255))

    # Ticks and labels
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
    except OSError:
        font = ImageFont.load_default()

    n_ticks = 6
    for i in range(n_ticks):
        frac = i / (n_ticks - 1)
        x = int(frac * (w - 1))
        dist = z_min + frac * (z_max - z_min)
        label = f"{dist:.1f} m"
        draw.line([(x, gh), (x, gh + _COLORBAR_TICK_H)], fill=(200, 200, 200, 255), width=1)
        bbox = draw.textbbox((0, 0), label, font=font)
        lw = bbox[2] - bbox[0]
        tx = max(0, min(x - lw // 2, w - lw))
        draw.text((tx, gh + _COLORBAR_TICK_H + 2), label, fill=(220, 220, 220, 255), font=font)

    # Title annotation: dot radius
    title = f"LiDAR depth  ·  dot r={dot_r}px  ·  near=red  far=blue"
    bbox = draw.textbbox((0, 0), title, font=font)
    draw.text(
        (w // 2 - (bbox[2] - bbox[0]) // 2, gh + _COLORBAR_TICK_H + 2),
        title,
        fill=(180, 180, 180, 200),
        font=font,
    )

    return img


def process_depth_file(npz: Path, out_dir: Path, img_shape: tuple[int, int], *, overwrite: bool, dot_r: int = 3) -> None:
    """Generate depth_layer.png + depth_colorbar.png from a _depths.npz sidecar."""
    frame_tag = npz.stem[: -len("_depths")]
    layer_path = out_dir / f"{frame_tag}_depth_layer.png"
    if not overwrite and layer_path.exists():
        return

    data = np.load(npz)
    u: np.ndarray = data["u"].astype(np.int32)
    v: np.ndarray = data["v"].astype(np.int32)
    z_m: np.ndarray = data["z_m"].astype(np.float32)

    if z_m.size == 0:
        return

    h, w = img_shape
    # Depth range: clip extreme outliers at 2nd/98th percentile for better contrast.
    z_lo = float(np.percentile(z_m, 2))
    z_hi = float(np.percentile(z_m, 98))
    if z_hi <= z_lo:
        z_hi = z_lo + 1.0
    t = np.clip((z_m - z_lo) / (z_hi - z_lo), 0.0, 1.0)
    colors = _turbo_rgb(t)  # (N, 3)

    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    # Paint dots with radius dot_r (filled circle via offset grid)
    offsets = [
        (dy, dx)
        for dy in range(-dot_r, dot_r + 1)
        for dx in range(-dot_r, dot_r + 1)
        if dy * dy + dx * dx <= dot_r * dot_r
    ]
    for dy, dx in offsets:
        vv = np.clip(v + dy, 0, h - 1)
        uu = np.clip(u + dx, 0, w - 1)
        rgba[vv, uu, :3] = colors
        rgba[vv, uu, 3] = 255

    out_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgba, "RGBA").save(layer_path)

    cb_path = out_dir / f"{frame_tag}_depth_colorbar.png"
    if overwrite or not cb_path.exists():
        _make_colorbar(z_lo, z_hi, dot_r).save(cb_path)


# ---------------------------------------------------------------------------
# Directory processing
# ---------------------------------------------------------------------------

def process_file(png: Path, out_dir: Path, *, overwrite: bool) -> None:
    suffix = None
    for s in _POINT_SUFFIXES | _BLEND_SUFFIXES:
        if png.stem.endswith(s):
            suffix = s
            frame_tag = png.stem[: -len(s) - 1]
            break
    if suffix is None:
        return

    if not overwrite and (out_dir / f"{frame_tag}_{suffix}_layer.png").exists():
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    rgb = np.asarray(Image.open(png).convert("RGB"), dtype=np.uint8)

    if suffix in _POINT_SUFFIXES:
        _process_point_overlay(rgb, out_dir, frame_tag, suffix, src_dir=png.parent, overwrite=overwrite)
    else:
        _process_blend_overlay(rgb, out_dir, frame_tag, suffix)


def process_overlays_dir(overlays_dir: Path, *, overwrite: bool) -> int:
    pngs = sorted(overlays_dir.glob("*.png"))
    if not pngs:
        return 0
    out_dir = overlays_dir.parent.parent / f"{overlays_dir.parent.name}_gimp_layers"

    # Infer image shape from the first PNG (needed for depth layer canvas size).
    sample_shape: tuple[int, int] | None = None
    try:
        with Image.open(pngs[0]) as im:
            sample_shape = (im.height, im.width)
    except Exception:
        pass

    it = tqdm(pngs, desc=str(overlays_dir.parent.name), unit="frame") if tqdm else pngs
    for p in it:
        process_file(p, out_dir, overwrite=overwrite)

    # Depth sidecar files
    npz_files = sorted(overlays_dir.glob("*_depths.npz"))
    if npz_files and sample_shape is not None:
        it2 = tqdm(npz_files, desc=f"{overlays_dir.parent.name} depth", unit="frame") if tqdm else npz_files
        for npz in it2:
            process_depth_file(npz, out_dir, sample_shape, overwrite=overwrite)

    return len(pngs)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--overlays-dir", type=Path)
    group.add_argument("--results-dir", type=Path, help="Walk all .../overlays/ dirs under this path.")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    dirs = [args.overlays_dir] if args.overlays_dir else sorted(args.results_dir.rglob("overlays"))

    total = 0
    for d in dirs:
        if not d.is_dir():
            continue
        n = process_overlays_dir(d, overwrite=args.overwrite)
        if n:
            print(f"  {d.parent.name}/overlays → gimp_layers/ ({n} files → base + lidar pairs)")
        total += n

    print(f"\nDone. {total} overlay files processed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
