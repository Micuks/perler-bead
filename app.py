#!/usr/bin/env python3
"""Perler bead pattern generator with CIEDE2000 color matching, dithering, and board comparison."""

import io
import os
import json
import time
import uuid
import threading
from flask import Flask, request, jsonify, send_from_directory
from PIL import Image
import numpy as np
import cv2
from skimage.color import rgb2lab, deltaE_ciede2000
from colors import BRANDS, COLORS, COLOR_RGB

app = Flask(__name__, static_folder="static")
app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024  # 20MB (photos can be large)

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

print(f"LUT built in {time.time() - t0:.1f}s")

BEAD_RGB_FOR_MATMUL = BEAD_RGB_NP.copy()
_bead_sq = np.sum(BEAD_RGB_FOR_MATMUL ** 2, axis=1, keepdims=True).T


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


def _rgb_nearest(pixels_flat):
    flat = pixels_flat.astype(np.float64)
    pixel_sq = np.sum(flat ** 2, axis=1, keepdims=True)
    distances = pixel_sq + _bead_sq - 2 * flat @ BEAD_RGB_FOR_MATMUL.T
    return np.argmin(distances, axis=1)


def image_to_pattern(image_bytes, width=100, height=100, brand="mard",
                     dither="none", mode="ciede2000"):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGBA")
    bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
    img = Image.alpha_composite(bg, img).convert("RGB")
    brand_idx = BRANDS.get(brand, BRANDS["mard"])["index"]

    if mode == "classic":
        img = img.resize((width, height), Image.LANCZOS)
        pixels = np.array(img)
        idx_grid = _rgb_nearest(pixels.reshape(-1, 3)).reshape(height, width)
    elif dither == "floyd-steinberg":
        img = img.resize((width, height), Image.BOX)
        idx_grid = _floyd_steinberg_dither(np.array(img), brand_idx)
    elif dither == "ordered":
        img = img.resize((width, height), Image.BOX)
        pixels = _ordered_dither(np.array(img))
        idx_grid = _lut_lookup_vectorized(pixels.reshape(-1, 3)).reshape(height, width)
    else:
        img = img.resize((width, height), Image.BOX)
        pixels = np.array(img)
        idx_grid = _lut_lookup_vectorized(pixels.reshape(-1, 3)).reshape(height, width)
        idx_grid = _denoise_isolated(idx_grid)

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
def compare_board(photo_bytes, corners, pattern_data):
    """Compare a photo of a physical bead board against the expected pattern.

    corners: [[x1,y1],[x2,y2],[x3,y3],[x4,y4]] in photo pixel coords (TL, TR, BR, BL)
    """
    pw, ph = pattern_data["width"], pattern_data["height"]
    brand = pattern_data.get("brand", "mard")
    brand_idx = BRANDS.get(brand, BRANDS["mard"])["index"]

    # Decode photo
    img_array = np.frombuffer(photo_bytes, dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)  # BGR
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Perspective transform: map corners to a rectangle
    cell_px = 20  # pixels per bead cell in the warped image
    dst_w, dst_h = pw * cell_px, ph * cell_px
    src_pts = np.array(corners, dtype=np.float32)
    dst_pts = np.array([[0, 0], [dst_w, 0], [dst_w, dst_h], [0, dst_h]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped = cv2.warpPerspective(img_rgb, M, (dst_w, dst_h))

    # Sample each cell: center 60% to avoid grid lines/gaps
    margin = int(cell_px * 0.2)
    grid = []
    total = pw * ph
    correct = 0
    wrong = 0
    missing = 0

    # Detect "empty" by checking if color is very close to board base color
    # We'll use a simple brightness threshold for "empty/transparent" base
    EMPTY_BRIGHTNESS_THRESHOLD = 240  # very bright = likely empty white base

    for y in range(ph):
        row = []
        for x in range(pw):
            x0 = x * cell_px + margin
            y0 = y * cell_px + margin
            x1 = (x + 1) * cell_px - margin
            y1 = (y + 1) * cell_px - margin
            cell = warped[y0:y1, x0:x1]
            avg_color = cell.mean(axis=(0, 1)).astype(np.uint8)

            # Check if empty
            brightness = int(avg_color[0]) * 299 + int(avg_color[1]) * 587 + int(avg_color[2]) * 114
            brightness //= 1000

            expected = pattern_data["pattern"][y][x]

            if brightness > EMPTY_BRIGHTNESS_THRESHOLD:
                # Likely empty
                status = "missing"
                missing += 1
                actual_code = ""
                actual_hex = ""
            else:
                # Match to nearest bead color
                idx = _lut_lookup(avg_color[0], avg_color[1], avg_color[2])
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
        "width": pw,
        "height": ph,
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
    if mode not in ("ciede2000", "classic"):
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

    photo_bytes = request.files["photo"].read()
    result = compare_board(photo_bytes, corners, pattern_data)
    return jsonify(result)


@app.route("/api/brands")
def brands():
    return jsonify({k: v["name"] for k, v in BRANDS.items()})


@app.route("/api/colors")
def api_colors():
    return jsonify([{"hex": c[0], "coco": c[1], "mard": c[2], "manman": c[3], "panpan": c[4], "mixiaowo": c[5]} for c in COLORS])


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8892, debug=False)
