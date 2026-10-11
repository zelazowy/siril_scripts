# SPDX-License-Identifier: GPL-3.0-only
# Unofficial port of Seti Astro Suite Pro by Franklin Marek and contributors.
# Adapted for Siril on 2026-10-11; see NOTICE.md for sources and changes.
# This program is free software under GNU GPL version 3.
# Distributed WITHOUT ANY WARRANTY; see the accompanying LICENSE.
# Algorithms Copyright Franklin Marek and SAS Pro contributors.
# Siril adaptation Copyright (C) 2026 zelazowy.
# Unofficial standalone port; see README.md and NOTICE.md.
from __future__ import annotations
import numpy as np
import math
from typing import Tuple, Optional, Callable, Sequence
import sep
cv2 = None
_sep_import_error = None
DEFAULT_AUTO_SIGMA_LADDER = (100., 50., 25., 12., 8., 5., 3.)
DEFAULT_AUTO_SIGMA_MIN_STARS = 1200
cached_star_sources = cached_flux_radii = _cached_shape = _cached_threshold = None

def preview_rgb(image, target=0.25):
    """Display-only linked RGB midtones stretch, fitted to this image.

    Use one black point and midtones parameter for RGB, so the preview
    does not independently neutralize the channels. Statistics are sampled;
    rendered pixels are not decimated. The returned array owns its data.
    """
    step = max(1, int(np.ceil(max(image.shape[:2]) / 1024)))
    sample = image[::step, ::step]
    med = np.median(sample, axis=(0, 1))
    mad = 1.4826 * np.median(np.abs(sample - med), axis=(0, 1))
    black = max(0.0, float(np.min(med - 2.8 * mad)))
    white = 1.0
    bg = float(np.clip((np.mean(med) - black) / max(white-black, 1e-8), 1e-6, 1-1e-6))
    m = float(np.clip(((target-1)*bg) / ((2*target-1)*bg-target), 1e-6, 1-1e-6))
    output = np.empty(image.shape, dtype=np.uint8)
    for y in range(0, image.shape[0], 256):
        values = np.clip((image[y:y+256] - black) / max(white-black, 1e-8), 0, 1)
        stretched = ((m-1)*values) / ((2*m-1)*values-m)
        output[y:y+256] = np.rint(np.clip(stretched, 0, 1)*255).astype(np.uint8)
    return output

def _to_float01(image):
    image = np.asarray(image)
    if image.dtype == np.uint16:
        image = image.astype(np.float32) / 65535.
    elif image.dtype == np.uint8:
        image = image.astype(np.float32) / 255.
    elif not np.issubdtype(image.dtype, np.floating):
        raise ValueError("Unsupported pixel type")
    if not np.isfinite(image).all():
        raise ValueError("Image contains NaN or infinite pixels")
    # Expand the nominal range only as needed. One shared affine transform
    # preserves channel relationships and avoids discarding extreme pixels.
    low = min(0.0, float(image.min()))
    high = max(1.0, float(image.max()))
    if low < 0.0 or high > 1.0:
        # Float64 intermediate also handles the full finite float32 range.
        image = (image.astype(np.float64) - low) / (high - low)
    return image.astype(np.float32, copy=True)

def _remove_channel_pedestal(img_rgb01: np.ndarray) -> np.ndarray:
    """
    Remove a per-channel pedestal using the whole image:
        out[...,c] = out[...,c] - min(out[...,c])
    Assumes float32-ish data; returns float32 clipped to [0,1].
    """
    out = img_rgb01.astype(np.float32, copy=True)

    mins = np.nanmin(out.reshape(-1, 3), axis=0).astype(np.float32)  # (3,)
    # If a channel is all-NaN, nanmin returns NaN; guard it:
    mins = np.where(np.isfinite(mins), mins, 0.0).astype(np.float32)

    out -= mins.reshape(1, 1, 3)
    return np.clip(out, 0.0, 1.0).astype(np.float32, copy=False)

def background_neutralize_rgb(
    img: np.ndarray,
    rect_xywh: tuple[int, int, int, int],
    mode: str = "pivot1",
    *,
    remove_pedestal: bool = True,
) -> np.ndarray:
    """
    ...
    Step 0 (optional): whole-image pedestal removal (per-channel)
    """
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError("Background Neutralization requires a 3-channel RGB image.")

    # Step 0: pedestal removal on the WHOLE image (optional)
    out = _remove_channel_pedestal(img) if remove_pedestal else img.astype(np.float32, copy=True)

    # Resolve sample rect (use pedestal-free image for medians)
    h, w, _ = out.shape
    x, y, rw, rh = rect_xywh
    x = max(0, min(int(x), w - 1))
    y = max(0, min(int(y), h - 1))
    rw = max(1, min(int(rw), w - x))
    rh = max(1, min(int(rh), h - y))

    sample = out[y:y + rh, x:x + rw, :]
    m = np.median(sample, axis=(0, 1)).astype(np.float32)  # (3,)
    t = float(np.mean(m))

    eps = 1e-8

    if mode == "offset":
        delta = (t - m).reshape(1, 1, 3)

        # cap deltas so we cannot clip
        ch_min = out.reshape(-1, 3).min(axis=0)
        ch_max = out.reshape(-1, 3).max(axis=0)
        delta = np.clip(
            delta,
            (-ch_min + 0.0).reshape(1, 1, 3),
            (1.0 - ch_max).reshape(1, 1, 3)
        )

        return np.clip(out + delta, 0.0, 1.0).astype(np.float32, copy=False)

    # pivot around 1.0 scaling (highlight-protect)
    denom = np.maximum(1.0 - m, eps)         # (3,)
    g = (1.0 - t) / denom                    # (3,)
    g = np.clip(g, 0.0, 10.0)                # sanity cap

    out = 1.0 - (1.0 - out) * g.reshape(1, 1, 3)
    return np.clip(out, 0.0, 1.0).astype(np.float32, copy=False)

def _find_best_patch_center(lum: np.ndarray) -> tuple[int, int]:
    """Port of your downhill-walk tile search (works on a luminance plane)."""
    h, w = lum.shape
    if min(h, w) < 10:
        return h // 2, w // 2
    rng = np.random.default_rng(0)
    th, tw = h // 10, w // 10
    
    # Optimized: compute 10x10 tile medians using strided views where possible
    # This avoids repeated slicing and is cache-friendlier
    meds = np.zeros((10, 10), dtype=np.float32)
    
    # For tiles that fit evenly, use reshape + median (faster than loop)
    crop_h, crop_w = th * 10, tw * 10
    if crop_h <= h and crop_w <= w:
        lum_crop = lum[:crop_h, :crop_w]
        # Reshape to (10, th, 10, tw) and compute medians
        tiles = lum_crop.reshape(10, th, 10, tw).transpose(0, 2, 1, 3).reshape(10, 10, -1)
        meds = np.median(tiles, axis=2).astype(np.float32)
        
        # Handle edge tiles if image doesn't divide evenly
        if h > crop_h or w > crop_w:
            # Bottom row edge
            if h > crop_h:
                for j in range(10):
                    x0, x1 = j * tw, (j + 1) * tw if j < 9 else w
                    meds[9, j] = np.median(lum[9*th:h, x0:x1])
            # Right column edge
            if w > crop_w:
                for i in range(10):
                    y0, y1 = i * th, (i + 1) * th if i < 9 else h
                    meds[i, 9] = np.median(lum[y0:y1, 9*tw:w])
    else:
        # Fallback for very small images
        for i in range(10):
            for j in range(10):
                y0, x0 = i * th, j * tw
                y1 = (i + 1) * th if i < 9 else h
                x1 = (j + 1) * tw if j < 9 else w
                meds[i, j] = np.median(lum[y0:y1, x0:x1])

    idxs = np.argsort(meds.flatten())[:2]

    finals = []
    for idx in idxs:
        ti, tj = divmod(int(idx), 10)
        y0, x0 = ti * th, tj * tw
        y1 = (ti + 1) * th if ti < 9 else h
        x1 = (tj + 1) * tw if tj < 9 else w
        for _ in range(200):
            y = rng.integers(y0, y1)
            x = rng.integers(x0, x1)
            while True:
                mv, mpos = lum[y, x], (y, x)
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        if dy == 0 and dx == 0:
                            continue
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < h and 0 <= nx < w and lum[ny, nx] < mv:
                            mv, mpos = lum[ny, nx], (ny, nx)
                if mpos == (y, x):
                    break
                y, x = mpos
            finals.append((y, x))

    best_val = np.inf
    best_pt = (h // 2, w // 2)
    for (y, x) in finals:
        y0 = max(0, y - 25); y1 = min(h, y + 25)
        x0 = max(0, x - 25); x1 = min(w, x + 25)
        m = np.median(lum[y0:y1, x0:x1])
        if m < best_val:
            best_val, best_pt = m, (y, x)
    return best_pt

def auto_rect_box(img_rgb: np.ndarray, box: int = 50, margin: int = 100) -> tuple[int, int, int, int]:
    """
    Find a robust box×box background rectangle (>= margin px margins) in image space.
    Returns (x, y, w, h).

    Notes:
      - img_rgb must be HxWx3 float/uint in any range (we only use relative luminance).
      - box is clamped so it always fits within the image + margins.
    """
    if img_rgb.ndim != 3 or img_rgb.shape[2] != 3:
        raise ValueError("Auto background finder expects a 3-channel RGB image.")

    H, W, _ = img_rgb.shape
    box = int(box)
    margin = min(margin, max(0, (min(H, W) - min(box, min(H, W))) // 2))
    if min(H, W) < 10:
        return (0, 0, W, H)

    # Clamp box so it fits inside the image after margins.
    # Ensure at least 10px and at least 1px interior.
    max_box_w = max(10, W - 2 * margin - 2)
    max_box_h = max(10, H - 2 * margin - 2)
    max_box = max(10, min(max_box_w, max_box_h))
    box = int(np.clip(box, 10, max_box))

    half = box // 2

    # Luminance proxy
    lum = img_rgb.mean(axis=2).astype(np.float32, copy=False)

    # Your existing routine (assumed to return (cy, cx) in image coords)
    cy, cx = _find_best_patch_center(lum)

    # Keep center far enough from edges so the full box fits
    min_cx, max_cx = margin + half, W - (margin + half)
    min_cy, max_cy = margin + half, H - (margin + half)
    cx = int(np.clip(cx, min_cx, max_cx))
    cy = int(np.clip(cy, min_cy, max_cy))

    # Refine around the center.
    # Step scales with box so the search is meaningful at different sizes.
    step = max(4, half // 2)  # e.g. 50->12, 80->20, 30->7
    best_val = np.inf
    ty, tx = cy, cx

    for dy in (-step, 0, +step):
        for dx in (-step, 0, +step):
            y = int(np.clip(cy + dy, min_cy, max_cy))
            x = int(np.clip(cx + dx, min_cx, max_cx))
            y0, y1 = y - half, y - half + box
            x0, x1 = x - half, x - half + box

            # Safety (should already be safe due to clamping, but keep it robust)
            if y0 < 0 or x0 < 0 or y1 > H or x1 > W:
                continue

            m = float(np.median(lum[y0:y1, x0:x1]))
            if m < best_val:
                best_val, ty, tx = m, y, x

    # Top-left anchored rect, exact box size
    x0 = int(tx - half)
    y0 = int(ty - half)

    # Final clamp (again, belt + suspenders)
    x0 = int(np.clip(x0, margin, W - margin - box))
    y0 = int(np.clip(y0, margin, H - margin - box))

    return (x0, y0, box, box)

def auto_rect_50x50(img_rgb: np.ndarray) -> tuple[int, int, int, int]:
    """Backward-compatible wrapper."""
    return auto_rect_box(img_rgb, box=50, margin=100)

def _sample_star_circle_medians(
    rgb: np.ndarray,
    sources: np.ndarray,
    radius: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Sample each detected star as the median RGB value inside a small circular
    aperture centered on the rounded star centroid.

    Returns
    -------
    samples : (N,3) float32
    keep_idx : (N,) int indices into the input `sources`
    """
    h, w = rgb.shape[:2]
    rr2 = int(radius) * int(radius)

    samples = []
    keep_idx = []

    for i in range(len(sources)):
        cx = int(round(float(sources["x"][i])))
        cy = int(round(float(sources["y"][i])))

        x0 = max(0, cx - radius)
        x1 = min(w, cx + radius + 1)
        y0 = max(0, cy - radius)
        y1 = min(h, cy + radius + 1)

        if x1 <= x0 or y1 <= y0:
            continue

        yy, xx = np.ogrid[y0:y1, x0:x1]
        mask = ((xx - cx) ** 2 + (yy - cy) ** 2) <= rr2
        if not np.any(mask):
            continue

        patch = rgb[y0:y1, x0:x1, :]
        vals = patch[mask]
        if vals.size == 0:
            continue

        med = np.median(vals, axis=0).astype(np.float32)
        if not np.all(np.isfinite(med)):
            continue

        samples.append(med)
        keep_idx.append(i)

    if len(samples) == 0:
        raise ValueError("No valid stellar samples available for white balance.")

    return np.asarray(samples, dtype=np.float32), np.asarray(keep_idx, dtype=np.int32)

def _spatially_sample_sources(sources: np.ndarray, r: np.ndarray,
                               img_shape: tuple, grid: int = 3,
                               per_cell: int = 500) -> tuple[np.ndarray, np.ndarray]:
    """
    Divide the image into a grid x grid spatial grid and keep at most
    per_cell stars per cell, preferring the brightest (highest flux).
    Returns filtered (sources, r).
    """
    H, W = img_shape[:2]
    cell_h = H / grid
    cell_w = W / grid

    keep = []
    for row in range(grid):
        for col in range(grid):
            y0 = row * cell_h
            y1 = (row + 1) * cell_h
            x0 = col * cell_w
            x1 = (col + 1) * cell_w

            in_cell = np.where(
                (sources["x"] >= x0) & (sources["x"] < x1) &
                (sources["y"] >= y0) & (sources["y"] < y1)
            )[0]

            if len(in_cell) == 0:
                continue

            if len(in_cell) <= per_cell:
                keep.append(in_cell)
            else:
                cell_flux = sources["flux"][in_cell]
                top = np.argpartition(cell_flux, -per_cell)[-per_cell:]
                keep.append(in_cell[top])

    if not keep:
        return sources, r

    idx = np.concatenate(keep)
    return sources[idx], r[idx]

def _waterfall_sep_extract(
    data_sub: np.ndarray,
    err_val: float,
    ladder: Sequence[float],
    min_stars: int,
    progress: Optional[Callable[[float, int, bool], None]] = None,
) -> tuple[np.ndarray, float]:
    """
    Run sep.extract against `data_sub` at each σ in `ladder` (highest first),
    returning the first result whose star count ≥ `min_stars`. If no rung
    clears the bar, the result at the lowest (most sensitive) σ is returned
    so callers still have something to work with on sparse fields.

    `progress`, when provided, is called with (sigma, count, settled) after
    every rung — `settled` is True on the one that is being returned.

    Returns
    -------
    sources : structured ndarray from sep.extract (may be empty on truly
              blank frames; callers must handle that case).
    used_sigma : the σ that produced `sources`.
    """
    if not ladder:
        raise ValueError("Auto-σ ladder is empty.")

    last_sources: Optional[np.ndarray] = None
    last_sigma: float = float(ladder[-1])

    for i, sig in enumerate(ladder):
        sig_f = float(sig)
        try:
            srcs = sep.extract(data_sub, sig_f, err=err_val)
        except Exception:
            # Treat a failure at this rung as "zero detections" and keep
            # descending — some rungs can blow the pixstack on dense fields
            srcs = np.empty(0, dtype=object)

        n = 0 if srcs is None else int(len(srcs))
        last_sources = srcs
        last_sigma = sig_f

        at_last = (i >= len(ladder) - 1)
        hit = (n >= int(min_stars))

        if progress is not None:
            try:
                progress(sig_f, n, bool(hit or at_last))
            except Exception:
                pass

        if hit:
            return srcs, sig_f

    # Ladder exhausted without hitting min_stars — fall back to the last try.
    return (last_sources if last_sources is not None
            else sep.extract(data_sub, float(ladder[-1]), err=err_val)), last_sigma

def apply_star_based_white_balance(
    image: np.ndarray,
    threshold: float = 1.5,
    autostretch: bool = False,
    reuse_cached_sources: bool = False,
    return_star_colors: bool = False,
    use_color_matrix: bool = False,       # ← new
    # ── Auto-σ waterfall ────────────────────────────────────────────────
    auto_sigma: bool = False,
    auto_sigma_min_stars: int = DEFAULT_AUTO_SIGMA_MIN_STARS,
    auto_sigma_ladder: Sequence[float] = DEFAULT_AUTO_SIGMA_LADDER,
    on_sigma_resolved: Optional[Callable[[float], None]] = None,
) -> Tuple[np.ndarray, int, np.ndarray, np.ndarray, np.ndarray] | Tuple[np.ndarray, int, np.ndarray]:
    """
    Star-based white balance using:
      1) shared Background Neutralization module
      2) SEP star detection on the BN image
      3) small circular median samples per star (radius=3 px)
      4) median-anchored white-point scaling
      5) NO second BN pass afterwards

    Parameters
    ----------
    image : np.ndarray
        RGB image (any dtype). Assumed RGB ordering.
    threshold : float
        SEP detection threshold (in background sigma). Ignored if
        `auto_sigma=True`.
    autostretch : bool
        If True, overlay is built from an autostretched BN view for visibility.
    reuse_cached_sources : bool
        Reuse detections only if image shape and threshold match. Disabled
        automatically when `auto_sigma=True` since the resolved σ isn't
        known until detection runs.
    return_star_colors : bool
        If True, also returns (raw_star_pixels, after_star_pixels).
    auto_sigma : bool
        If True, run the σ waterfall described in `auto_sigma_ladder`,
        picking the highest σ that yields ≥ `auto_sigma_min_stars` sources.
        The BN pass and sep.Background estimation are performed exactly
        once — only sep.extract itself is retried per rung — so the extra
        cost is modest even on big frames.
    auto_sigma_min_stars : int
        Target star count that qualifies a σ rung as "good enough".
    auto_sigma_ladder : Sequence[float]
        σ values tried in order (highest first).
    on_sigma_resolved : callable(float) | None
        Optional callback invoked with the σ that was ultimately used.
        Useful for UI code that wants to drive a slider to the auto value.

    Returns
    -------
    balanced_rgb : float32 RGB in [0,1]
    star_count   : int
    overlay_rgb  : float32 RGB in [0,1] with star ellipses drawn
    (optional) raw_star_pixels   : (N,3) float32
    (optional) after_star_pixels : (N,3) float32
    """
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("apply_star_based_white_balance: input must be an RGB image (H,W,3).")

    if use_color_matrix:
        raise ValueError("Advanced color matrix is not included in this port")
    if sep is None:
        raise ImportError(
            "apply_star_based_white_balance requires the 'sep' package. "
            f"Import error was: {_sep_import_error!r}"
        )

    img_rgb = _to_float01(image).astype(np.float32, copy=False)

    # ------------------------------------------------------------------
    # 1) Shared background neutralization
    # ------------------------------------------------------------------
    rect = auto_rect_50x50(img_rgb)
    bg_neutral = background_neutralize_rgb(img_rgb, rect).astype(np.float32, copy=False)

    pivot = np.median(bg_neutral[rect[1]:rect[1]+rect[3], rect[0]:rect[0]+rect[2], :],
                      axis=(0, 1)).astype(np.float32)

    # ------------------------------------------------------------------
    # 2) Detect stars on BN luminance
    # ------------------------------------------------------------------
    gray = np.mean(bg_neutral, axis=2).astype(np.float32, copy=False)
    bkg = sep.Background(gray)
    data_sub = gray - bkg.back()
    err_val = float(bkg.globalrms)

    global cached_star_sources, cached_flux_radii, _cached_shape, _cached_threshold

    # Cache logic doesn't apply when we're auto-picking σ — the "key" would
    # be the resolved σ, which isn't known yet.
    use_cache = (
        (not auto_sigma)
        and bool(reuse_cached_sources)
        and cached_star_sources is not None
        and cached_flux_radii is not None
        and _cached_shape == tuple(img_rgb.shape)
        and _cached_threshold is not None
        and abs(float(_cached_threshold) - float(threshold)) < 1e-9
    )

    if use_cache:
        sources = cached_star_sources
        r = cached_flux_radii
        used_threshold = float(threshold)
    else:
        if auto_sigma:
            # σ waterfall — BN already done, so this only re-runs sep.extract
            sources, used_threshold = _waterfall_sep_extract(
                data_sub, err_val,
                ladder=auto_sigma_ladder,
                min_stars=auto_sigma_min_stars,
            )
        else:
            used_threshold = float(threshold)
            sources = sep.extract(data_sub, used_threshold, err=err_val)

        if sources is None or len(sources) == 0:
            raise ValueError("No sources detected for Star-Based White Balance.")

        r, _ = sep.flux_radius(
            gray,
            sources["x"], sources["y"],
            2.0 * sources["a"], 0.2,
            normflux=sources["flux"],
            subpix=5
        )

        # Keep only star-like detections
        mask = (r > 0) & (r <= 10)
        sources = sources[mask]
        r = r[mask]

        if len(sources) == 0:
            raise ValueError("All detected sources were rejected as non-stellar (too large).")

        # Spatially uniform sampling — caps total stars on dense fields
        sources, r = _spatially_sample_sources(sources, r, img_rgb.shape, grid=3, per_cell=500)

        # Cache the already-filtered result (keyed by whatever σ actually ran)
        cached_star_sources = sources
        cached_flux_radii = r
        _cached_shape = tuple(img_rgb.shape)
        _cached_threshold = float(used_threshold)

    # Report the σ we actually used — UI code can use this to drive a slider
    if on_sigma_resolved is not None:
        try:
            on_sigma_resolved(float(used_threshold))
        except Exception:
            pass

    # ------------------------------------------------------------------
    # 3) Sample stars as circular medians, not center pixels
    # ------------------------------------------------------------------
    raw_star_pixels, keep_idx = _sample_star_circle_medians(img_rgb, sources, radius=3)
    sources = sources[keep_idx]
    r = r[keep_idx]

    bn_star_pixels, _ = _sample_star_circle_medians(bg_neutral, sources, radius=3)

    if len(bn_star_pixels) == 0:
        raise ValueError("No stellar samples remained after circular median sampling.")

    # ------------------------------------------------------------------
    # 4) Build preview overlay
    # ------------------------------------------------------------------
    disp = bg_neutral.copy()

    if cv2 is not None:
        overlay_bgr = cv2.cvtColor((disp * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        for i in range(len(sources)):
            cx = float(sources["x"][i])
            cy = float(sources["y"][i])
            a = float(sources["a"][i])
            b = float(sources["b"][i])
            theta_deg = float(sources["theta"][i] * 180.0 / np.pi)

            center = (int(round(cx)), int(round(cy)))
            axes = (max(1, int(round(3 * a))), max(1, int(round(3 * b))))

            cv2.ellipse(
                overlay_bgr,
                center,
                axes,
                angle=theta_deg,
                startAngle=0,
                endAngle=360,
                color=(0, 0, 255),
                thickness=1
            )

        overlay_rgb = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    else:
        overlay_rgb = disp.astype(np.float32, copy=False)

    # ------------------------------------------------------------------
    # 5) WB math: shift center to neutral + tilt toward blackbody slope
    # ------------------------------------------------------------------
    # ---- Step 1: measure current star color ratios ----
    eps = 1e-8
    ref_color = np.median(bn_star_pixels, axis=0).astype(np.float32)
    ref_color = np.where(ref_color <= eps, eps, ref_color)

    rb_stars = bn_star_pixels[:, 0] / (bn_star_pixels[:, 2] + eps)
    gb_stars = bn_star_pixels[:, 1] / (bn_star_pixels[:, 2] + eps)

    finite_mask = (
        np.isfinite(rb_stars) & np.isfinite(gb_stars) &
        (rb_stars > 0.1) & (gb_stars > 0.1) &
        (rb_stars < 4.0) & (gb_stars < 4.0)
    )

    TARGET_SLOPE = 0.55
    TILT_STRENGTH = 0.7

    if np.sum(finite_mask) >= 10:
        rb_f = rb_stars[finite_mask]
        gb_f = gb_stars[finite_mask]
        A = np.stack([rb_f, np.ones_like(rb_f)], axis=1)
        result = np.linalg.lstsq(A, gb_f, rcond=None)
        current_slope = float(np.clip(result[0][0], 0.05, 3.0))
    else:
        current_slope = TARGET_SLOPE

    # ---- Step 2: compute slope correction gain ratio ----
    # new_slope = (kg/kr) * current_slope = TARGET_SLOPE
    # So kg/kr = TARGET_SLOPE / current_slope
    # Split symmetrically: kr = 1/sqrt(ratio), kg = sqrt(ratio)
    slope_ratio = float(np.clip(TARGET_SLOPE / current_slope, 0.3, 3.0))
    blended_ratio = 1.0 + TILT_STRENGTH * (slope_ratio - 1.0)

    kr_tilt = 1.0 / math.sqrt(blended_ratio)
    kg_tilt = math.sqrt(blended_ratio)
    kb_tilt = 1.0

    # ---- Step 3: apply tilt to star samples, then find neutral shift ----
    # We want the TILTED median to land at neutral (equal RGB)
    # So compute where the median ends up after tilt, then correct
    tilt = np.array([kr_tilt, kg_tilt, kb_tilt], dtype=np.float32)
    tilted_median = ref_color * tilt
    tilted_median = np.where(tilted_median <= eps, eps, tilted_median)

    # Neutral shift: scale so max channel = all channels (gray)
    neutral_target = float(np.max(tilted_median))
    neutral_scale = neutral_target / tilted_median  # per-channel

    # ---- Final combined scaling: tilt * neutral (computed together, not sequentially) ----
    final_scaling = (tilt * neutral_scale).astype(np.float32)

    m = pivot.reshape((1, 1, 3)).astype(np.float32)
    g = final_scaling.reshape((1, 1, 3)).astype(np.float32)

    balanced = (bg_neutral.astype(np.float32) - m) * g + m
    balanced = np.clip(balanced, 0.0, 1.0).astype(np.float32, copy=False)
    # ------------------------------------------------------------------
    # 6) After-WB diagnostic samples, same circular-median method
    # ------------------------------------------------------------------
    after_star_pixels, _ = _sample_star_circle_medians(balanced, sources, radius=3)

    star_count = int(len(bn_star_pixels))

    if return_star_colors:
        return (
            balanced,
            star_count,
            overlay_rgb.astype(np.float32, copy=False),
            np.asarray(raw_star_pixels, dtype=np.float32),
            np.asarray(after_star_pixels, dtype=np.float32),
        )

    return (
        balanced,
        star_count,
        overlay_rgb.astype(np.float32, copy=False),
    )
