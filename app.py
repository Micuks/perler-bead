#!/usr/bin/env python3
"""Perler bead pattern generator - upload an image, get a bead pattern."""

import io
import json
import math
from flask import Flask, request, jsonify, send_from_directory
from PIL import Image
import numpy as np

app = Flask(__name__, static_folder="static")
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10MB

# Standard perler bead colors (name, hex, RGB)
BEAD_COLORS = [
    ("White", "#FFFFFF", (255, 255, 255)),
    ("Cream", "#E8D8B8", (232, 216, 184)),
    ("Yellow", "#FFD900", (255, 217, 0)),
    ("Orange", "#FF8C00", (255, 140, 0)),
    ("Red", "#DC143C", (220, 20, 60)),
    ("Bubblegum", "#FF69B4", (255, 105, 180)),
    ("Purple", "#8B008B", (139, 0, 139)),
    ("Dark Blue", "#00008B", (0, 0, 139)),
    ("Blue", "#1E90FF", (30, 144, 255)),
    ("Light Blue", "#87CEEB", (135, 206, 235)),
    ("Cyan", "#00CED1", (0, 206, 209)),
    ("Green", "#228B22", (34, 139, 34)),
    ("Light Green", "#90EE90", (144, 238, 144)),
    ("Lime", "#ADFF2F", (173, 255, 47)),
    ("Brown", "#8B4513", (139, 69, 19)),
    ("Tan", "#D2B48C", (210, 180, 140)),
    ("Gray", "#808080", (128, 128, 128)),
    ("Light Gray", "#C0C0C0", (192, 192, 192)),
    ("Charcoal", "#36454F", (54, 69, 79)),
    ("Black", "#000000", (0, 0, 0)),
    ("Peach", "#FFDAB9", (255, 218, 185)),
    ("Salmon", "#FA8072", (250, 128, 114)),
    ("Magenta", "#FF00FF", (255, 0, 255)),
    ("Lavender", "#E6E6FA", (230, 230, 250)),
    ("Plum", "#DDA0DD", (221, 160, 221)),
    ("Sky Blue", "#87CEFA", (135, 206, 250)),
    ("Teal", "#008080", (0, 128, 128)),
    ("Olive", "#6B8E23", (107, 142, 35)),
    ("Gold", "#FFD700", (255, 215, 0)),
    ("Rust", "#B7410E", (183, 65, 14)),
    ("Maroon", "#800000", (128, 0, 0)),
    ("Navy", "#000080", (0, 0, 128)),
    ("Forest", "#013220", (1, 50, 32)),
    ("Coral", "#FF7F50", (255, 127, 80)),
    ("Turquoise", "#40E0D0", (64, 224, 208)),
    ("Rose", "#FF007F", (255, 0, 127)),
    ("Periwinkle", "#CCCCFF", (204, 204, 255)),
    ("Mint", "#98FF98", (152, 255, 152)),
    ("Khaki", "#C3B091", (195, 176, 145)),
    ("Sand", "#C2B280", (194, 178, 128)),
]

BEAD_RGB = np.array([c[2] for c in BEAD_COLORS], dtype=np.float64)


def find_nearest_bead(rgb):
    """Find the nearest bead color using Euclidean distance in RGB space."""
    diff = BEAD_RGB - np.array(rgb, dtype=np.float64)
    distances = np.sum(diff ** 2, axis=1)
    idx = int(np.argmin(distances))
    return idx


def image_to_pattern(image_bytes, width=29, height=29):
    """Convert an uploaded image to a perler bead pattern."""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGBA")

    # Handle transparency: composite on white background
    bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
    img = Image.alpha_composite(bg, img).convert("RGB")

    img = img.resize((width, height), Image.LANCZOS)
    pixels = np.array(img)

    pattern = []
    color_counts = {}
    for y in range(height):
        row = []
        for x in range(width):
            r, g, b = int(pixels[y, x, 0]), int(pixels[y, x, 1]), int(pixels[y, x, 2])
            idx = find_nearest_bead((r, g, b))
            name, hex_color, _ = BEAD_COLORS[idx]
            row.append({"color": hex_color, "name": name})
            color_counts[name] = color_counts.get(name, 0) + 1
        pattern.append(row)

    color_summary = [{"name": k, "count": v, "hex": next(c[1] for c in BEAD_COLORS if c[0] == k)}
                     for k, v in sorted(color_counts.items(), key=lambda x: -x[1])]

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

    width = request.form.get("width", 29, type=int)
    height = request.form.get("height", 29, type=int)
    width = max(5, min(100, width))
    height = max(5, min(100, height))

    image_bytes = file.read()
    if not image_bytes:
        return jsonify({"error": "Empty image data"}), 400
    try:
        result = image_to_pattern(image_bytes, width, height)
    except Exception as e:
        return jsonify({"error": f"Failed to process image: {e}"}), 400
    return jsonify(result)


@app.route("/api/colors")
def colors():
    return jsonify([{"name": c[0], "hex": c[1]} for c in BEAD_COLORS])


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8892, debug=False)
