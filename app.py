#!/usr/bin/env python3
"""Perler bead pattern generator with CIEDE2000 color matching, dithering, and board comparison."""

import io
import os
import json
import time
import uuid
import threading
from flask import Flask, request, jsonify, send_from_directory
from flask_compress import Compress
from PIL import Image
import numpy as np
import cv2
from skimage.color import rgb2lab, deltaE_ciede2000
from colors import BRANDS, COLORS, COLOR_RGB

app = Flask(__name__, static_folder="static")
app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024  # 20MB (photos can be large)
Compress(app)

PATTERNS_DIR = os.path.join(os.path.dirname(__file__), "patterns")
os.makedirs(PATTERNS_DIR, exist_ok=True)
PATTERN_TTL = 7 * 24 * 3600  # 7 days

# --- Precompute CIEDE2000 lookup table at startup ---
LUT_BITS = 5
LUT_SIZE = 1 << LUT_BITS
LUT_SHIFT = 8 - LUT_BITS

BEAD_RGB_NP = np.array(COLOR_RGB, dtype=np.float64)
_bead_rgb_01 = BEAD_RGB_NP / 255.0
_bead_lab = rgb2lab(_bead_rgb_01.reshape(1, -1, 3)).reshape(-1, 3)

print(f"Building CIEDE2000 LUT ({LUT_SIZE}^3 = {LUT_SIZE**3} entries)...")
t0 = time.time()

_r = np.arange(LUT_SIZE) * (255 / (LUT_SIZE - 1))
_grid = np.stack(np.meshgrid(_r, _r, _r, indexing='ij'), axis=-1).reshape(-1, 3)
_grid_01 = _grid / 255.0
_grid_lab = rgb2lab(_grid_01.reshape(1, -1, 3)).reshape(-1, 3)

N_BEADS = len(COLORS)
N_LUT = len(_grid_lab)
CIEDE2000_LUT = np.empty(N_LUT, dtype=np.int16)

CHUNK = 1024
for i in range(0, N_LUT, CHUNK):
    end = min(i + CHUNK, N_LUT)
    chunk_lab = _grid_lab[i:end]
    dists = np.empty((end - i, N_BEADS), dtype=np.float64)
    for j in range(N_BEADS):
        bead_tile = np.tile(_bead_lab[j], (end - i, 1))
        dists[:, j] = deltaE_ciede2000(chunk_lab, bead_tile, channel_axis=-1)
    CIEDE2000_LUT[i:end] = np.argmin(dists, axis=1).astype(np.int16)

print(f"CIEDE2000 LUT built in {time.time() - t0:.1f}s")

# Build RGB Euclidean LUT (for classic mode) — much faster to compute
t1 = time.time()
_grid_f = _grid.astype(np.float64)
_bead_f = BEAD_RGB_NP.astype(np.float64)
_grid_sq = np.sum(_grid_f ** 2, axis=1, keepdims=True)  # (N_LUT, 1)
_bead_sq_lut = np.sum(_bead_f ** 2, axis=1, keepdims=True).T  # (1, N_BEADS)
_rgb_dists = _grid_sq + _bead_sq_lut - 2 * _grid_f @ _bead_f.T  # (N_LUT, N_BEADS)
RGB_LUT = np.argmin(_rgb_dists, axis=1).astype(np.int16)
del _grid_f, _grid_sq, _bead_sq_lut, _rgb_dists
print(f"RGB LUT built in {time.time() - t1:.3f}s")


# --- Pattern storage with TTL ---
def _save_pattern(pattern_data):
    pid = uuid.uuid4().hex[:12]
    path = os.path.join(PATTERNS_DIR, f"{pid}.json")
    record = {"data": pattern_data, "created": time.time()}
    with open(path, "w") as f:
        json.dump(record, f)
    return pid


def _load_pattern(pid):
    path = os.path.join(PATTERNS_DIR, f"{pid}.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        record = json.load(f)
    if time.time() - record["created"] > PATTERN_TTL:
        os.remove(path)
        return None
    return record["data"]


def _cleanup_patterns():
    """Remove expired patterns."""
    now = time.time()
    for fname in os.listdir(PATTERNS_DIR):
        if not fname.endswith(".json"):
            continue
        path = os.path.join(PATTERNS_DIR, fname)
        try:
            with open(path) as f:
                record = json.load(f)
            if now - record.get("created", 0) > PATTERN_TTL:
                os.remove(path)
        except Exception:
            pass


def _periodic_cleanup():
    while True:
        time.sleep(3600)
        _cleanup_patterns()


threading.Thread(target=_periodic_cleanup, daemon=True).start()


# --- Color matching functions ---
def _lut_lookup(r, g, b):
    ri = int(r) >> LUT_SHIFT
    gi = int(g) >> LUT_SHIFT
    bi = int(b) >> LUT_SHIFT
    return int(CIEDE2000_LUT[ri * LUT_SIZE * LUT_SIZE + gi * LUT_SIZE + bi])


def _lut_lookup_vectorized(pixels_flat):
    quantized = pixels_flat.astype(np.int32) >> LUT_SHIFT
    lut_indices = quantized[:, 0] * LUT_SIZE * LUT_SIZE + quantized[:, 1] * LUT_SIZE + quantized[:, 2]
    return CIEDE2000_LUT[lut_indices]


def _rgb_lut_lookup_vectorized(pixels_flat):
    quantized = pixels_flat.astype(np.int32) >> LUT_SHIFT
    lut_indices = quantized[:, 0] * LUT_SIZE * LUT_SIZE + quantized[:, 1] * LUT_SIZE + quantized[:, 2]
    return RGB_LUT[lut_indices]


def _floyd_steinberg_dither(pixels, brand_idx):
    h, w = pixels.shape[:2]
    buf = pixels.astype(np.float64)
    result = np.empty((h, w), dtype=np.int32)
    for y in range(h):
        for x in range(w):
            old = np.clip(buf[y, x], 0, 255)
            idx = _lut_lookup(old[0], old[1], old[2])
            result[y, x] = idx
            bead_rgb = BEAD_RGB_NP[idx]
            err = old - bead_rgb
            if x + 1 < w:
                buf[y, x + 1] += err * (7 / 16)
            if y + 1 < h:
                if x > 0:
                    buf[y + 1, x - 1] += err * (3 / 16)
                buf[y + 1, x] += err * (5 / 16)
                if x + 1 < w:
                    buf[y + 1, x + 1] += err * (1 / 16)
    return result


BAYER_4 = np.array([
    [0, 8, 2, 10], [12, 4, 14, 6],
    [3, 11, 1, 9], [15, 7, 13, 5],
], dtype=np.float64) / 16.0 - 0.5
ORDERED_SPREAD = 48.0


def _ordered_dither(pixels):
    h, w = pixels.shape[:2]
    buf = pixels.astype(np.float64)
    by = np.tile(BAYER_4, (h // 4 + 1, w // 4 + 1))[:h, :w]
    for c in range(3):
        buf[:, :, c] += by * ORDERED_SPREAD
    return np.clip(buf, 0, 255).astype(np.uint8)


def _denoise_isolated(idx_grid):
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
                if any(n == cur for n in neighbors):
                    continue
                diff = BEAD_RGB_NP[cur] - BEAD_RGB_NP[dominant]
                dist_sq = float(np.dot(diff, diff))
                if dist_sq < CONTRAST_THRESHOLD_SQ:
                    out[y, x] = dominant
    return out



def image_to_pattern(image_bytes, width=100, height=100, brand="mard",
                     dither="none", mode="ciede2000"):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGBA")
    bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
    img = Image.alpha_composite(bg, img).convert("RGB")
    brand_idx = BRANDS.get(brand, BRANDS["mard"])["index"]

    if mode == "pixeloe":
        # PixelOE: contrast-aware outline expansion + adaptive downsampling
        import torch
        from pixeloe.torch.pixelize import pixelize as pixelize_torch
        from pixeloe.torch.utils import pre_resize, to_numpy
        target_size = max(width, height)
        patch_size = 4
        img_t = pre_resize(img, target_size=target_size, patch_size=patch_size)
        with torch.no_grad():
            result_t = pixelize_torch(img_t, pixel_size=patch_size, thickness=2,
                                      do_quant=False, no_post_upscale=True)
        pixeloe_img = Image.fromarray(to_numpy(result_t)[0])
        # Resize to exact target dimensions (PixelOE output may differ slightly)
        pixeloe_img = pixeloe_img.resize((width, height), Image.NEAREST)
        pixels = np.array(pixeloe_img)
        idx_grid = _lut_lookup_vectorized(pixels.reshape(-1, 3)).reshape(height, width)
    elif mode == "classic":
        # Lanczos preserves sharp edges/thin lines; RGB LUT for fast matching
        img = img.resize((width, height), Image.LANCZOS)
        pixels = np.array(img)
        idx_grid = _rgb_lut_lookup_vectorized(pixels.reshape(-1, 3)).reshape(height, width)
    elif dither == "floyd-steinberg":
        img = img.resize((width, height), Image.BOX)
        idx_grid = _floyd_steinberg_dither(np.array(img), brand_idx)
    elif dither == "ordered":
        img = img.resize((width, height), Image.BOX)
        pixels = _ordered_dither(np.array(img))
        idx_grid = _lut_lookup_vectorized(pixels.reshape(-1, 3)).reshape(height, width)
    else:
        # Edge-aware pipeline: resize to 4x → bilateral filter → BOX to target → CIEDE2000
        # Bilateral at 4x target resolution is fast and effective: smooths within-region
        # noise while preserving edges, so BOX resize produces fewer mixed colors.
        mid_w, mid_h = width * 4, height * 4
        img_mid = img.resize((mid_w, mid_h), Image.BOX)
        pixels_bgr = cv2.cvtColor(np.array(img_mid), cv2.COLOR_RGB2BGR)
        filtered_bgr = cv2.bilateralFilter(pixels_bgr, d=9, sigmaColor=75, sigmaSpace=75)
        filtered_rgb = cv2.cvtColor(filtered_bgr, cv2.COLOR_BGR2RGB)
        img_filtered = Image.fromarray(filtered_rgb)
        img_final = img_filtered.resize((width, height), Image.BOX)
        pixels = np.array(img_final)
        idx_grid = _lut_lookup_vectorized(pixels.reshape(-1, 3)).reshape(height, width)

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
    return {"pattern": pattern, "width": width, "height": height,
            "colors": color_summary, "brand": brand}


# --- Board comparison ---

# Precompute bead Lab values for matching
_bead_lab_flat = _bead_lab.copy()  # (N_BEADS, 3) in Lab


def _sample_grid(warped, grid_w, grid_h, cell_px):
    """Sample average color for each cell in the warped image. Returns (grid_h, grid_w, 3) uint8."""
    margin = int(cell_px * 0.2)
    sampled = np.empty((grid_h, grid_w, 3), dtype=np.uint8)
    for y in range(grid_h):
        for x in range(grid_w):
            x0 = x * cell_px + margin
            y0 = y * cell_px + margin
            x1 = (x + 1) * cell_px - margin
            y1 = (y + 1) * cell_px - margin
            sampled[y, x] = warped[y0:y1, x0:x1].mean(axis=(0, 1)).astype(np.uint8)
    return sampled


def _white_balance(sampled, empty_threshold=240):
    """Auto white-balance using bright (empty) cells as white reference."""
    brightness = (sampled[:, :, 0].astype(int) * 299 +
                  sampled[:, :, 1].astype(int) * 587 +
                  sampled[:, :, 2].astype(int) * 114) // 1000
    bright_mask = brightness > empty_threshold
    if bright_mask.sum() < 3:
        # Not enough empty cells — try using the top 10% brightest
        threshold = np.percentile(brightness, 90)
        bright_mask = brightness > threshold
    if bright_mask.sum() < 1:
        return sampled  # can't calibrate

    ref_white = sampled[bright_mask].mean(axis=0).astype(np.float64)
    # Avoid division by zero
    ref_white = np.maximum(ref_white, 1.0)
    scale = 255.0 / ref_white
    # Clamp scale to avoid extreme corrections
    scale = np.clip(scale, 0.5, 2.0)
    corrected = np.clip(sampled.astype(np.float64) * scale, 0, 255).astype(np.uint8)
    return corrected


def _build_ref_index_grid(pattern_data):
    """Convert pattern data to a 2D array of bead color indices."""
    ph, pw = pattern_data["height"], pattern_data["width"]
    # Map hex -> index
    hex_to_idx = {c[0]: i for i, c in enumerate(COLORS)}
    grid = np.empty((ph, pw), dtype=np.int32)
    for y in range(ph):
        for x in range(pw):
            h = pattern_data["pattern"][y][x]["color"]
            grid[y, x] = hex_to_idx.get(h, 0)
    return grid


def _auto_align(query_indices, ref_indices):
    """Find best (dy, dx) offset of query within reference using soft Lab distance.

    Returns (best_dy, best_dx, best_score).
    """
    qh, qw = query_indices.shape
    rh, rw = ref_indices.shape
    if qh > rh or qw > rw:
        return 0, 0, float('inf')

    # Precompute Lab values
    query_lab = _bead_lab_flat[query_indices]  # (qh, qw, 3)
    ref_lab = _bead_lab_flat[ref_indices]      # (rh, rw, 3)

    best_score = float('inf')
    best_dy, best_dx = 0, 0

    for dy in range(rh - qh + 1):
        for dx in range(rw - qw + 1):
            window = ref_lab[dy:dy + qh, dx:dx + qw]  # (qh, qw, 3)
            diff = window - query_lab
            score = float(np.sum(diff * diff))  # sum of squared Lab distances
            if score < best_score:
                best_score = score
                best_dy, best_dx = dy, dx

    return best_dy, best_dx, best_score


def compare_board(photo_bytes, corners, pattern_data, grid_w=None, grid_h=None):
    """Compare a photo of a physical bead board against the expected pattern.

    corners: [[x1,y1],[x2,y2],[x3,y3],[x4,y4]] in photo pixel coords (TL, TR, BR, BL)
    grid_w, grid_h: number of beads in the photo (if None, assume full pattern)
    """
    pw, ph = pattern_data["width"], pattern_data["height"]
    # If grid dimensions not specified, assume full pattern
    if not grid_w:
        grid_w = pw
    if not grid_h:
        grid_h = ph
    # Clamp to pattern size
    grid_w = min(grid_w, pw)
    grid_h = min(grid_h, ph)

    brand = pattern_data.get("brand", "mard")
    brand_idx = BRANDS.get(brand, BRANDS["mard"])["index"]

    # Decode & perspective transform
    img_array = np.frombuffer(photo_bytes, dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    cell_px = 20
    dst_w, dst_h = grid_w * cell_px, grid_h * cell_px
    src_pts = np.array(corners, dtype=np.float32)
    dst_pts = np.array([[0, 0], [dst_w, 0], [dst_w, dst_h], [0, dst_h]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped = cv2.warpPerspective(img_rgb, M, (dst_w, dst_h))

    # Sample & white-balance
    sampled = _sample_grid(warped, grid_w, grid_h, cell_px)
    sampled = _white_balance(sampled)

    # Quantize each cell
    query_indices = _lut_lookup_vectorized(sampled.reshape(-1, 3)).reshape(grid_h, grid_w).astype(np.int32)

    # Auto-align: find best position within the full pattern
    ref_indices = _build_ref_index_grid(pattern_data)

    if grid_w == pw and grid_h == ph:
        # Full pattern, no alignment needed
        offset_y, offset_x = 0, 0
    else:
        offset_y, offset_x, _ = _auto_align(query_indices, ref_indices)

    # Compare at the matched position
    EMPTY_BRIGHTNESS_THRESHOLD = 240
    grid = []
    correct = wrong = missing = 0
    total = grid_h * grid_w

    for y in range(grid_h):
        row = []
        for x in range(grid_w):
            ry, rx = offset_y + y, offset_x + x
            expected = pattern_data["pattern"][ry][rx]
            avg = sampled[y, x]
            brightness = (int(avg[0]) * 299 + int(avg[1]) * 587 + int(avg[2]) * 114) // 1000

            if brightness > EMPTY_BRIGHTNESS_THRESHOLD:
                status = "missing"
                missing += 1
                actual_code = ""
                actual_hex = ""
            else:
                idx = int(query_indices[y, x])
                entry = COLORS[idx]
                actual_code = entry[brand_idx]
                actual_hex = entry[0]
                if actual_code == expected["code"]:
                    status = "correct"
                    correct += 1
                else:
                    status = "wrong"
                    wrong += 1

            row.append({
                "status": status,
                "actual_code": actual_code,
                "actual_hex": actual_hex,
                "expected_code": expected["code"],
                "expected_hex": expected["color"],
            })
        grid.append(row)

    return {
        "grid": grid,
        "width": grid_w,
        "height": grid_h,
        "offset_x": offset_x,
        "offset_y": offset_y,
        "pattern_width": pw,
        "pattern_height": ph,
        "stats": {
            "total": total,
            "correct": correct,
            "wrong": wrong,
            "missing": missing,
            "completion_pct": round(correct / total * 100, 1) if total > 0 else 0,
        }
    }


# --- Routes ---
@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/compare")
def compare_page():
    return send_from_directory("static", "compare.html")


@app.route("/colors")
def colors_page():
    return send_from_directory("static", "colors.html")


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
    if mode not in ("ciede2000", "classic", "pixeloe"):
        mode = "ciede2000"

    image_bytes = file.read()
    result = image_to_pattern(image_bytes, width, height, brand, dither, mode)

    # Auto-save pattern
    pid = _save_pattern(result)
    result["pattern_id"] = pid

    return jsonify(result)


@app.route("/api/pattern/<pid>")
def get_pattern(pid):
    if not pid.isalnum() or len(pid) > 20:
        return jsonify({"error": "Invalid pattern ID"}), 400
    data = _load_pattern(pid)
    if data is None:
        return jsonify({"error": "Pattern not found or expired"}), 404
    data["pattern_id"] = pid
    return jsonify(data)


@app.route("/api/compare", methods=["POST"])
def compare():
    if "photo" not in request.files:
        return jsonify({"error": "No photo uploaded"}), 400
    corners_str = request.form.get("corners")
    if not corners_str:
        return jsonify({"error": "No corners provided"}), 400
    corners = json.loads(corners_str)
    if len(corners) != 4:
        return jsonify({"error": "Need exactly 4 corners"}), 400

    # Load pattern: by ID or from uploaded data
    pid = request.form.get("pattern_id")
    pattern_json = request.form.get("pattern_data")
    if pid:
        pattern_data = _load_pattern(pid)
        if pattern_data is None:
            return jsonify({"error": "Pattern not found or expired"}), 404
    elif pattern_json:
        pattern_data = json.loads(pattern_json)
    else:
        return jsonify({"error": "No pattern_id or pattern_data provided"}), 400

    grid_w = request.form.get("grid_w", type=int)
    grid_h = request.form.get("grid_h", type=int)

    photo_bytes = request.files["photo"].read()
    result = compare_board(photo_bytes, corners, pattern_data, grid_w, grid_h)
    return jsonify(result)


@app.route("/api/brands")
def brands():
    return jsonify({k: v["name"] for k, v in BRANDS.items()})


@app.route("/api/colors")
def api_colors():
    return jsonify([{"hex": c[0], "coco": c[1], "mard": c[2], "manman": c[3], "panpan": c[4], "mixiaowo": c[5]} for c in COLORS])


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8892, debug=False, threaded=True)
