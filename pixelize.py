"""Perceptual pixelization for fixed-palette bead patterns.

Implements a simplified Gerstner et al. 2012 "Pixelated Image Abstraction" pipeline:
joint EM optimization of superpixel-to-cell assignment and soft palette commitment,
with saliency-weighted M-step. The palette is fixed (no k-means split), so the
algorithm reduces to alternating SLIC-style assignment + soft palette annealing.

Also exports a Lab-space Floyd-Steinberg dither that fixes the RGB/Lab metric
mismatch in the legacy implementation.
"""

from __future__ import annotations

import cv2
import numpy as np
from skimage.color import rgb2lab, lab2rgb


# --- Tunables (Gerstner-style; T values in Lab² units) ---
EM_ITERS = 8
T_INIT = 50.0
T_MIN = 5.0
SLIC_M = 45.0          # SLIC compactness, in Lab units
EARLY_STOP_DELTA = 0.5  # Lab-unit max-change threshold for early stop


def _prepare_mid(rgb: np.ndarray, out_w: int, out_h: int) -> np.ndarray:
    """Resize input RGB to a target ~8x output, with floor 2x to avoid degeneracy."""
    H_in, W_in = rgb.shape[:2]
    mid_w = max(2 * out_w, min(8 * out_w, W_in))
    mid_h = max(2 * out_h, min(8 * out_h, H_in))
    if (mid_w, mid_h) == (W_in, H_in):
        return rgb
    interp = cv2.INTER_AREA if mid_w * mid_h < W_in * H_in else cv2.INTER_LINEAR
    return cv2.resize(rgb, (mid_w, mid_h), interpolation=interp)


def _compute_saliency(L: np.ndarray) -> np.ndarray:
    """Sobel-magnitude saliency on the L channel, floored to 0.2 so flat regions still count."""
    L8 = np.clip(L * 2.55, 0, 255).astype(np.uint8)  # Lab L is 0..100
    gx = cv2.Sobel(L8, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(L8, cv2.CV_32F, 0, 1, ksize=3)
    sal = np.sqrt(gx * gx + gy * gy)
    sal /= (sal.max() + 1e-6)
    return (0.2 + 0.8 * sal).astype(np.float32)


def _e_step(lab_in: np.ndarray, cell_lab: np.ndarray, cell_pos_y: np.ndarray,
            cell_pos_x: np.ndarray, mid_h: int, mid_w: int, H: int, W: int,
            search_radius: int) -> np.ndarray:
    """Vectorized SLIC-style assignment with bounded candidate window.

    Returns flat cell index per input pixel: shape (mid_h, mid_w) int32.
    """
    yy, xx = np.meshgrid(np.arange(mid_h, dtype=np.int32),
                         np.arange(mid_w, dtype=np.int32), indexing='ij')
    gy = (yy * H) // mid_h
    gx = (xx * W) // mid_w

    # SLIC compactness weight: M^2 / S^2 with S^2 = cell area in input pixels.
    S2 = (mid_h / H) * (mid_w / W)
    spatial_w = (SLIC_M * SLIC_M) / max(S2, 1.0)

    yy_f = yy.astype(np.float32)
    xx_f = xx.astype(np.float32)

    best_d = np.full((mid_h, mid_w), np.inf, dtype=np.float32)
    best_k = np.zeros((mid_h, mid_w), dtype=np.int32)

    for dy in range(-search_radius, search_radius + 1):
        for dx in range(-search_radius, search_radius + 1):
            cy = np.clip(gy + dy, 0, H - 1)
            cx = np.clip(gx + dx, 0, W - 1)
            cand_lab = cell_lab[cy, cx]                 # (mid_h, mid_w, 3)
            d_lab2 = ((lab_in - cand_lab) ** 2).sum(axis=-1)
            d_xy2 = (yy_f - cell_pos_y[cy, cx]) ** 2 + (xx_f - cell_pos_x[cy, cx]) ** 2
            D = d_lab2 + spatial_w * d_xy2
            flat_k = (cy * W + cx).astype(np.int32)
            mask = D < best_d
            best_d = np.where(mask, D, best_d)
            best_k = np.where(mask, flat_k, best_k)

    return best_k


def _m_step(lab_in: np.ndarray, sal: np.ndarray, assign: np.ndarray,
            prev_cell_lab: np.ndarray, H: int, W: int) -> np.ndarray:
    """Saliency-weighted mean Lab per cell. Empty cells inherit prev_cell_lab."""
    flat = assign.ravel()
    sal_flat = sal.ravel()
    weight = np.bincount(flat, weights=sal_flat, minlength=H * W).astype(np.float32)
    sums = np.empty((H * W, 3), dtype=np.float32)
    for c in range(3):
        sums[:, c] = np.bincount(
            flat,
            weights=(sal_flat * lab_in[..., c].ravel()).astype(np.float64),
            minlength=H * W,
        ).astype(np.float32)
    denom = np.maximum(weight, 1e-6)
    new_lab = (sums / denom[:, None]).reshape(H, W, 3)
    empty = (weight == 0).reshape(H, W)
    if empty.any():
        new_lab[empty] = prev_cell_lab[empty]
    return new_lab


def _soft_palette_pull(cell_lab: np.ndarray, bead_lab: np.ndarray, T: float) -> np.ndarray:
    """Soft-assign each cell to the palette under ΔE76²/T softmax, return new cell_lab.

    Uses the ‖a-b‖² = ‖a‖²+‖b‖²−2 a·b identity to avoid an (H*W, K, 3) temp buffer.
    """
    H, W, _ = cell_lab.shape
    flat = cell_lab.reshape(-1, 3).astype(np.float32)
    bead_lab32 = bead_lab.astype(np.float32)
    flat_sq = (flat * flat).sum(-1, keepdims=True)              # (H*W, 1)
    bead_sq = (bead_lab32 * bead_lab32).sum(-1, keepdims=True).T  # (1, K)
    d2 = flat_sq + bead_sq - 2.0 * flat @ bead_lab32.T          # (H*W, K)
    d2 -= d2.min(axis=-1, keepdims=True)
    p = np.exp(-d2 / max(T, 1e-3))
    p /= p.sum(axis=-1, keepdims=True) + 1e-12
    return (p @ bead_lab32).reshape(H, W, 3)


def perceptual_pixelize(rgb: np.ndarray, out_w: int, out_h: int, *,
                         bead_lab: np.ndarray, lut: np.ndarray | None = None,
                         lut_bits: int = 5,
                         iters: int = EM_ITERS,
                         T_init: float = T_INIT, T_min: float = T_MIN
                         ) -> np.ndarray:
    """Gerstner-style perceptual pixelization.

    rgb:      (H_in, W_in, 3) uint8
    out_w/h:  output bead grid dimensions
    bead_lab: (K, 3) float — palette in Lab
    lut:      optional 5-bit RGB-indexed CIEDE2000 LUT for final hard quantization.
              If None, falls back to ΔE76 argmin in Lab.

    Returns: (out_h, out_w) int16 — palette indices.
    """
    H, W = int(out_h), int(out_w)
    rgb_mid = _prepare_mid(rgb, W, H)
    mid_h, mid_w = rgb_mid.shape[:2]

    lab_in = rgb2lab(rgb_mid / 255.0).astype(np.float32)
    sal = _compute_saliency(lab_in[..., 0])

    # Cell centers in mid-resolution pixel coords, (H, W) float32 each.
    cy_idx, cx_idx = np.meshgrid(np.arange(H, dtype=np.float32),
                                 np.arange(W, dtype=np.float32), indexing='ij')
    cell_pos_y = (cy_idx + 0.5) * (mid_h / H)
    cell_pos_x = (cx_idx + 0.5) * (mid_w / W)

    # Bilinear sample to seed cell_lab.
    sy = np.clip(cell_pos_y, 0, mid_h - 1)
    sx = np.clip(cell_pos_x, 0, mid_w - 1)
    map_x = sx.astype(np.float32)
    map_y = sy.astype(np.float32)
    cell_lab = cv2.remap(lab_in, map_x, map_y, interpolation=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_REPLICATE).astype(np.float32)

    # Tiny outputs: widen search window so the candidate set covers all cells.
    search_radius = 2 if min(H, W) <= 8 else 1

    T_schedule = np.linspace(T_init, T_min, iters, dtype=np.float32)
    prev_lab = cell_lab.copy()

    for it in range(iters):
        assign = _e_step(lab_in, cell_lab, cell_pos_y, cell_pos_x,
                         mid_h, mid_w, H, W, search_radius)
        cell_lab = _m_step(lab_in, sal, assign, cell_lab, H, W)
        # Within-cell smoothing: tiny bilateral on the (H, W, 3) cell grid.
        # Lab-space sigmaColor=8 ≈ 2× JND; spatial sigma 1.2 stops at 1-cell radius.
        if min(H, W) >= 5:
            cell_lab = cv2.bilateralFilter(cell_lab, d=5, sigmaColor=8.0, sigmaSpace=1.2)
        cell_lab = _soft_palette_pull(cell_lab, bead_lab, float(T_schedule[it]))

        delta = float(np.max(np.abs(cell_lab - prev_lab)))
        if delta < EARLY_STOP_DELTA:
            break
        prev_lab = cell_lab.copy()

    # Final hard quantization. Reuse the precomputed CIEDE2000 RGB LUT if provided
    # (5-bit RGB quantization is < 3 ΔE error; cell_lab here lies near palette so fine).
    rgb_back = np.clip(lab2rgb(cell_lab.reshape(1, -1, 3))[0] * 255.0, 0, 255).astype(np.uint8)
    if lut is not None:
        shift = 8 - lut_bits
        size = 1 << lut_bits
        q = rgb_back.astype(np.int32) >> shift
        idx_lut = q[:, 0] * size * size + q[:, 1] * size + q[:, 2]
        idx = lut[idx_lut].astype(np.int16)
    else:
        # ΔE76 argmin fallback
        flat = cell_lab.reshape(-1, 3).astype(np.float32)
        bead = bead_lab.astype(np.float32)
        flat_sq = (flat * flat).sum(-1, keepdims=True)
        bead_sq = (bead * bead).sum(-1, keepdims=True).T
        d2 = flat_sq + bead_sq - 2.0 * flat @ bead.T
        idx = np.argmin(d2, axis=-1).astype(np.int16)
    return idx.reshape(H, W)


def floyd_steinberg_lab(rgb: np.ndarray, out_w: int, out_h: int, *,
                         bead_lab: np.ndarray) -> np.ndarray:
    """Floyd-Steinberg error diffusion fully in Lab space.

    Fixes the legacy bug where errors were computed in RGB but matched in Lab —
    error direction was inconsistent with the matching metric.

    rgb:      (H_in, W_in, 3) uint8
    bead_lab: (K, 3) Lab palette
    Returns:  (out_h, out_w) int16
    """
    H, W = int(out_h), int(out_w)
    rgb_small = cv2.resize(rgb, (W, H), interpolation=cv2.INTER_AREA)
    buf = rgb2lab(rgb_small / 255.0).astype(np.float32)        # (H, W, 3)
    bead = bead_lab.astype(np.float32)                          # (K, 3)
    bead_sq = (bead * bead).sum(-1)                             # (K,)
    result = np.empty((H, W), dtype=np.int16)

    for y in range(H):
        for x in range(W):
            cur = buf[y, x]
            # ΔE76 argmin: ‖cur-bead‖² = ‖cur‖² + ‖bead‖² − 2 cur·bead
            # The ‖cur‖² term is constant across beads, so skip it for argmin.
            d2 = bead_sq - 2.0 * (bead @ cur)
            k = int(np.argmin(d2))
            result[y, x] = k
            err = cur - bead[k]
            if x + 1 < W:
                buf[y, x + 1] += err * (7.0 / 16.0)
            if y + 1 < H:
                if x > 0:
                    buf[y + 1, x - 1] += err * (3.0 / 16.0)
                buf[y + 1, x] += err * (5.0 / 16.0)
                if x + 1 < W:
                    buf[y + 1, x + 1] += err * (1.0 / 16.0)
    return result
