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

    brand_idx = BRANDS.get(brand, BRANDS["coco"])["index"]

    pattern = []
    color_counts = {}
    for y in range(height):
        row = []
        for x in range(width):
            r, g, b = int(pixels[y, x, 0]), int(pixels[y, x, 1]), int(pixels[y, x, 2])
            diff = BEAD_RGB - np.array([r, g, b], dtype=np.float64)
            idx = int(np.argmin(np.sum(diff ** 2, axis=1)))
            entry = COLORS[idx]
            code = entry[brand_idx]
            hex_color = entry[0]
            row.append({"color": hex_color, "code": code})
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
