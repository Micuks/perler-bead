#!/usr/bin/env python3
"""Perler bead pattern generator with CIEDE2000 color matching and dithering."""

import io
import time
from flask import Flask, request, jsonify, send_from_directory
from PIL import Image
import numpy as np
from skimage.color import rgb2lab, deltaE_ciede2000
from colors import BRANDS, COLORS, COLOR_RGB

app = Flask(__name__, static_folder="static")
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10MB

# --- Precompute CIEDE2000 lookup table at startup ---
# Quantize RGB to 32 levels per channel -> 32^3 = 32768 entries
LUT_BITS = 5  # 2^5 = 32 levels
LUT_SIZE = 1 << LUT_BITS  # 32
LUT_SHIFT = 8 - LUT_BITS  # 3

# Bead colors in LAB space
BEAD_RGB_NP = np.array(COLOR_RGB, dtype=np.float64)
_bead_rgb_01 = BEAD_RGB_NP / 255.0
_bead_lab = rgb2lab(_bead_rgb_01.reshape(1, -1, 3)).reshape(-1, 3)

print(f"Building CIEDE2000 LUT ({LUT_SIZE}^3 = {LUT_SIZE**3} entries)...")
t0 = time.time()

# Generate all quantized RGB values
_r = np.arange(LUT_SIZE) * (255 / (LUT_SIZE - 1))
_grid = np.stack(np.meshgrid(_r, _r, _r, indexing='ij'), axis=-1).reshape(-1, 3)
_grid_01 = _grid / 255.0
_grid_lab = rgb2lab(_grid_01.reshape(1, -1, 3)).reshape(-1, 3)

# Compute CIEDE2000 distance from each LUT entry to each bead color
# Process in chunks to manage memory
N_BEADS = len(COLORS)
N_LUT = len(_grid_lab)
CIEDE2000_LUT = np.empty(N_LUT, dtype=np.int16)

CHUNK = 1024
for i in range(0, N_LUT, CHUNK):
    end = min(i + CHUNK, N_LUT)
    chunk_lab = _grid_lab[i:end]  # (chunk, 3)
    dists = np.empty((end - i, N_BEADS), dtype=np.float64)
    for j in range(N_BEADS):
        # Broadcast bead_lab[j] to match chunk shape
        bead_tile = np.tile(_bead_lab[j], (end - i, 1))  # (chunk, 3)
        dists[:, j] = deltaE_ciede2000(chunk_lab, bead_tile, channel_axis=-1)
    CIEDE2000_LUT[i:end] = np.argmin(dists, axis=1).astype(np.int16)

print(f"LUT built in {time.time() - t0:.1f}s")

# Also keep a simple RGB fallback for speed
BEAD_RGB_FOR_MATMUL = BEAD_RGB_NP.copy()
_bead_sq = np.sum(BEAD_RGB_FOR_MATMUL ** 2, axis=1, keepdims=True).T  # (1, N)


def _lut_lookup(r, g, b):
    """Look up nearest bead index via precomputed CIEDE2000 LUT."""
    ri = int(r) >> LUT_SHIFT
    gi = int(g) >> LUT_SHIFT
    bi = int(b) >> LUT_SHIFT
    return int(CIEDE2000_LUT[ri * LUT_SIZE * LUT_SIZE + gi * LUT_SIZE + bi])


def _lut_lookup_vectorized(pixels_flat):
    """Vectorized LUT lookup for (N,3) uint8 array."""
    quantized = pixels_flat.astype(np.int32) >> LUT_SHIFT
    lut_indices = quantized[:, 0] * LUT_SIZE * LUT_SIZE + quantized[:, 1] * LUT_SIZE + quantized[:, 2]
    return CIEDE2000_LUT[lut_indices]


def _floyd_steinberg_dither(pixels, brand_idx):
    """Floyd-Steinberg error diffusion dithering with CIEDE2000 matching."""
    h, w = pixels.shape[:2]
    # Work in float to accumulate error
    buf = pixels.astype(np.float64)
    result = np.empty((h, w), dtype=np.int32)

    for y in range(h):
        for x in range(w):
            # Clamp current pixel
            old = np.clip(buf[y, x], 0, 255)
            # Find nearest bead via LUT
            idx = _lut_lookup(old[0], old[1], old[2])
            result[y, x] = idx
            # Compute error
            bead_rgb = BEAD_RGB_NP[idx]
            err = old - bead_rgb
            # Distribute error
            if x + 1 < w:
                buf[y, x + 1] += err * (7 / 16)
            if y + 1 < h:
                if x > 0:
                    buf[y + 1, x - 1] += err * (3 / 16)
                buf[y + 1, x] += err * (5 / 16)
                if x + 1 < w:
                    buf[y + 1, x + 1] += err * (1 / 16)

    return result


# Bayer 4x4 ordered dithering matrix
BAYER_4 = np.array([
    [0, 8, 2, 10],
    [12, 4, 14, 6],
    [3, 11, 1, 9],
    [15, 7, 13, 5],
], dtype=np.float64) / 16.0 - 0.5  # center around 0, range [-0.5, 0.5)

# Scale factor for ordered dithering spread
ORDERED_SPREAD = 48.0  # how much to perturb pixel values


def _ordered_dither(pixels):
    """Apply Bayer ordered dithering, return perturbed pixel array."""
    h, w = pixels.shape[:2]
    buf = pixels.astype(np.float64)
    # Tile bayer matrix across image
    by = np.tile(BAYER_4, (h // 4 + 1, w // 4 + 1))[:h, :w]
    # Add threshold to all channels
    for c in range(3):
        buf[:, :, c] += by * ORDERED_SPREAD
    return np.clip(buf, 0, 255).astype(np.uint8)


def _denoise_isolated(idx_grid):
    """Remove isolated single-pixel outliers caused by Lanczos edge bleeding.

    Only replaces a pixel when ALL of:
    1. 3+ of its 4-connected neighbors agree on a different color
    2. The pixel's color is SIMILAR to the dominant neighbor color
       (low RGB distance = likely a Lanczos blending artifact)

    High-contrast isolated pixels (e.g. dark eyes on light face) are kept
    because they are intentional features, not blending artifacts.
    """
    # Lanczos blending artifacts produce colors close to the true edge colors.
    # Real features (eyes, highlights) are drastically different from surroundings.
    # Threshold: squared RGB distance. ~3000 ≈ delta of ~30 per channel.
    CONTRAST_THRESHOLD_SQ = 3000

    h, w = idx_grid.shape
    out = idx_grid.copy()
    for y in range(h):
        for x in range(w):
            cur = idx_grid[y, x]
            neighbors = []
            if y > 0:     neighbors.append(idx_grid[y - 1, x])
            if y < h - 1: neighbors.append(idx_grid[y + 1, x])
            if x > 0:     neighbors.append(idx_grid[y, x - 1])
            if x < w - 1: neighbors.append(idx_grid[y, x + 1])
            if len(neighbors) < 3:
                continue
            counts = {}
            for n in neighbors:
                counts[n] = counts.get(n, 0) + 1
            dominant = max(counts, key=counts.get)
            if dominant != cur and counts[dominant] >= 3:
                # Keep if the pixel has ANY same-color neighbor (part of a line/feature)
                if any(n == cur for n in neighbors):
                    continue
                # Truly isolated (zero same-color neighbors):
                # only remove if low contrast (blending artifact, not a real detail)
                diff = BEAD_RGB_NP[cur] - BEAD_RGB_NP[dominant]
                dist_sq = float(np.dot(diff, diff))
                if dist_sq < CONTRAST_THRESHOLD_SQ:
                    out[y, x] = dominant
    return out


def _rgb_nearest(pixels_flat):
    """Original RGB Euclidean nearest-neighbor matching (vectorized matmul)."""
    flat = pixels_flat.astype(np.float64)
    pixel_sq = np.sum(flat ** 2, axis=1, keepdims=True)
    distances = pixel_sq + _bead_sq - 2 * flat @ BEAD_RGB_FOR_MATMUL.T
    return np.argmin(distances, axis=1)


def image_to_pattern(image_bytes, width=100, height=100, brand="mard",
                     dither="none", mode="ciede2000"):
    """Convert an uploaded image to a fuse bead pattern."""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGBA")
    bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
    img = Image.alpha_composite(bg, img).convert("RGB")

    brand_idx = BRANDS.get(brand, BRANDS["mard"])["index"]

    if mode == "classic":
        # Classic pipeline: Lanczos (sharp) + RGB nearest-neighbor, no denoise
        img = img.resize((width, height), Image.LANCZOS)
        pixels = np.array(img)
        flat = pixels.reshape(-1, 3)
        idx_grid = _rgb_nearest(flat).reshape(height, width)
    elif dither == "floyd-steinberg":
        img = img.resize((width, height), Image.BOX)
        pixels = np.array(img)
        idx_grid = _floyd_steinberg_dither(pixels, brand_idx)
    elif dither == "ordered":
        img = img.resize((width, height), Image.BOX)
        pixels = _ordered_dither(np.array(img))
        flat = pixels.reshape(-1, 3)
        idx_grid = _lut_lookup_vectorized(flat).reshape(height, width)
    else:
        # CIEDE2000 pipeline: BOX (no ringing) + perceptual matching + denoise
        img = img.resize((width, height), Image.BOX)
        pixels = np.array(img)
        flat = pixels.reshape(-1, 3)
        idx_grid = _lut_lookup_vectorized(flat).reshape(height, width)
        idx_grid = _denoise_isolated(idx_grid)

    # Build pattern JSON
    pattern = []
    color_counts = {}
    for y in range(height):
        row = []
        for x in range(width):
            idx = int(idx_grid[y, x])
            entry = COLORS[idx]
            row.append({"color": entry[0], "code": entry[brand_idx]})
            color_counts[idx] = color_counts.get(idx, 0) + 1
        pattern.append(row)

    color_summary = [
        {"code": COLORS[idx][brand_idx], "hex": COLORS[idx][0], "count": count}
        for idx, count in sorted(color_counts.items(), key=lambda x: -x[1])
    ]

    return {"pattern": pattern, "width": width, "height": height, "colors": color_summary}


@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/api/generate", methods=["POST"])
def generate():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    width = request.form.get("width", 100, type=int)
    height = request.form.get("height", 100, type=int)
    width = max(5, min(200, width))
    height = max(5, min(200, height))
    brand = request.form.get("brand", "mard")
    if brand not in BRANDS:
        brand = "mard"
    dither = request.form.get("dither", "none")
    if dither not in ("none", "floyd-steinberg", "ordered"):
        dither = "none"
    mode = request.form.get("mode", "ciede2000")
    if mode not in ("ciede2000", "classic"):
        mode = "ciede2000"

    image_bytes = file.read()
    result = image_to_pattern(image_bytes, width, height, brand, dither, mode)
    return jsonify(result)


@app.route("/api/brands")
def brands():
    return jsonify({k: v["name"] for k, v in BRANDS.items()})


@app.route("/api/colors")
def api_colors():
    return jsonify([{"hex": c[0], "coco": c[1], "mard": c[2], "manman": c[3], "panpan": c[4], "mixiaowo": c[5]} for c in COLORS])


@app.route("/colors")
def colors_page():
    return send_from_directory("static", "colors.html")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8892, debug=False)
