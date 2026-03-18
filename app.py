#!/usr/bin/env python3
"""Perler bead pattern generator - upload an image, get a bead pattern."""

import io
from flask import Flask, request, jsonify, send_from_directory
from PIL import Image
import numpy as np
from colors import BRANDS, COLORS, COLOR_RGB

app = Flask(__name__, static_folder="static")
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10MB

BEAD_RGB = np.array(COLOR_RGB, dtype=np.float64)


def image_to_pattern(image_bytes, width=100, height=100, brand="mard"):
    """Convert an uploaded image to a fuse bead pattern."""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGBA")
    bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
    img = Image.alpha_composite(bg, img).convert("RGB")
    img = img.resize((width, height), Image.LANCZOS)
    pixels = np.array(img)

    brand_idx = BRANDS.get(brand, BRANDS["mard"])["index"]

    # Vectorized nearest-color matching: (H*W, 3) vs (N, 3)
    flat = pixels.reshape(-1, 3).astype(np.float64)  # (H*W, 3)
    # Compute squared distances to all bead colors at once
    # Using broadcasting: (H*W, 1, 3) - (1, N, 3) -> (H*W, N, 3) -> sum -> (H*W, N)
    # Memory-efficient: compute in chunks if needed, but 10k x 205 is fine
    diffs = flat[:, np.newaxis, :] - BEAD_RGB[np.newaxis, :, :]
    distances = np.sum(diffs ** 2, axis=2)  # (H*W, N)
    indices = np.argmin(distances, axis=1)  # (H*W,)

    # Build pattern
    pattern = []
    color_counts = {}
    idx_grid = indices.reshape(height, width)
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
        brand = "coco"

    image_bytes = file.read()
    result = image_to_pattern(image_bytes, width, height, brand)
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
