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

# Artkal C-series fuse bead colors (code, name, hex, RGB)
BEAD_COLORS = [
    ("C01", "White", "#EAEEF3", (234, 238, 243)),
    ("C02", "Black", "#292A2B", (41, 42, 43)),
    ("C03", "Tangerine", "#FFA630", (255, 166, 48)),
    ("C04", "Yellow Orange", "#E68739", (230, 135, 57)),
    ("C05", "Tall Poppy", "#CB3531", (203, 53, 49)),
    ("C06", "Red", "#B61927", (182, 25, 39)),
    ("C07", "Carnation Pink", "#E182B0", (225, 130, 176)),
    ("C08", "Hot Pink", "#DC519A", (220, 81, 154)),
    ("C09", "Magenta", "#DA4383", (218, 67, 131)),
    ("C10", "Picasso", "#EADE7F", (234, 222, 127)),
    ("C11", "Yellow", "#EAC125", (234, 193, 37)),
    ("C12", "Pistachio", "#97CF87", (151, 207, 135)),
    ("C13", "Pastel Green", "#8BB23A", (139, 178, 58)),
    ("C14", "Green", "#009053", (0, 144, 83)),
    ("C15", "Green Tea", "#00765F", (0, 118, 95)),
    ("C16", "Bright Carrot", "#F96F40", (249, 111, 64)),
    ("C17", "Orange", "#EB6027", (235, 96, 39)),
    ("C18", "Sky Blue", "#A7CDDE", (167, 205, 222)),
    ("C19", "Baby Blue", "#2EABD8", (46, 171, 216)),
    ("C20", "Light Blue", "#0084CE", (0, 132, 206)),
    ("C21", "Dark Blue", "#004FA4", (0, 79, 164)),
    ("C22", "Bubble Gum", "#F2BFB8", (242, 191, 184)),
    ("C23", "Sand", "#DCA384", (220, 163, 132)),
    ("C24", "Beeswax", "#EED39E", (238, 211, 158)),
    ("C25", "Lavender", "#8A7EC2", (138, 126, 194)),
    ("C26", "Pastel Lavender", "#9165B2", (145, 101, 178)),
    ("C27", "Purple", "#48337E", (72, 51, 126)),
    ("C28", "Marigold", "#B27938", (178, 121, 56)),
    ("C29", "Buccaneer", "#B35540", (179, 85, 64)),
    ("C30", "Redwood", "#9A4541", (154, 69, 65)),
    ("C31", "Light Brown", "#895D49", (137, 93, 73)),
    ("C32", "Brown", "#65463D", (101, 70, 61)),
    ("C33", "Gray", "#959698", (149, 150, 152)),
    ("C34", "Dark Gray", "#70757B", (112, 117, 123)),
    ("C35", "Silver", "#676B73", (103, 107, 115)),
    ("C36", "Old Pink", "#CE6D83", (206, 109, 131)),
    ("C37", "True Blue", "#0078BF", (0, 120, 191)),
    ("C38", "Turquoise", "#55A4D9", (85, 164, 217)),
    ("C39", "Shadow Green", "#9EC9CD", (158, 201, 205)),
    ("C40", "Key Lemon Pie", "#CDC03F", (205, 192, 63)),
    ("C41", "Pastel Yellow", "#E1D367", (225, 211, 103)),
    ("C42", "Sandstorm", "#E1C835", (225, 200, 53)),
    ("C43", "Paprika", "#B11836", (177, 24, 54)),
    ("C44", "Burning Sand", "#EE927C", (238, 146, 124)),
    ("C45", "Spring Green", "#5DDB5D", (93, 219, 93)),
    ("C46", "Canary", "#E2E65D", (226, 230, 93)),
    ("C47", "Vanilla", "#E9C1A6", (233, 193, 166)),
    ("C48", "Corn", "#ECC03D", (236, 192, 61)),
    ("C49", "Raspberry Pink", "#EF67B2", (239, 103, 178)),
    ("C50", "Maverick", "#C0B7D7", (192, 183, 215)),
    ("C51", "Spring Sun", "#DFDABD", (223, 218, 189)),
    ("C52", "Butterfly Bush", "#4F3989", (79, 57, 137)),
    ("C53", "Bright Green", "#8EC324", (142, 195, 36)),
    ("C54", "Medium Turquoise", "#00A5A1", (0, 165, 161)),
    ("C55", "Conifer", "#6CC24A", (108, 194, 74)),
    ("C56", "Oslo Gray", "#82878B", (130, 135, 139)),
    ("C57", "Fresh Red", "#BC0423", (188, 4, 35)),
    ("C58", "Black Rock", "#36384D", (54, 56, 77)),
    ("C59", "Scarlett", "#531A23", (83, 26, 35)),
    ("C60", "Sea Mist", "#B2D7CE", (178, 215, 206)),
    ("C61", "Feta", "#F1EB9C", (241, 235, 156)),
    ("C62", "Carnation", "#FC3F3F", (252, 63, 63)),
    ("C63", "Pink Pearl", "#EABEDB", (234, 190, 219)),
    ("C64", "Rose", "#A50050", (165, 0, 80)),
    ("C65", "Mango", "#EF7E2E", (239, 126, 46)),
    ("C66", "Wild Watermelon", "#FC6C85", (252, 108, 133)),
    ("C67", "Orchid", "#B14EB5", (177, 78, 181)),
    ("C68", "Toothpaste Blue", "#69C2EE", (105, 194, 238)),
    ("C69", "Mine Shaft", "#383E44", (56, 62, 68)),
    ("C70", "Brunswick Green", "#153838", (21, 56, 56)),
    ("C71", "Goldenrod", "#E8AE00", (232, 174, 0)),
    ("C72", "Pastel Orange", "#D9B35E", (217, 179, 94)),
    ("C73", "Sienna", "#BB6833", (187, 104, 51)),
    ("C74", "Deer", "#CDB277", (205, 178, 119)),
    ("C75", "Clay", "#AA744E", (170, 116, 78)),
    ("C76", "Coral Red", "#EC625E", (236, 98, 94)),
    ("C77", "Deep Chestnut", "#BE5D65", (190, 93, 101)),
    ("C78", "Red Wine", "#99323A", (153, 50, 58)),
    ("C79", "Light Sea Blue", "#68C4D2", (104, 196, 210)),
    ("C80", "Sea Blue", "#0093A9", (0, 147, 169)),
    ("C81", "Steel Blue", "#5AB0BF", (90, 176, 191)),
    ("C82", "Azure", "#009EC2", (0, 158, 194)),
    ("C83", "Dark Steel Blue", "#0084B2", (0, 132, 178)),
    ("C84", "Dark Algae", "#ADAD29", (173, 173, 41)),
    ("C85", "Dark Olive", "#8F8E3C", (143, 142, 60)),
    ("C86", "Jade Green", "#007D2B", (0, 125, 43)),
    ("C87", "Ghost White", "#D4D8D3", (212, 216, 211)),
    ("C88", "Ash Grey", "#C2C4C2", (194, 196, 194)),
    ("C89", "Light Gray", "#A7ACAD", (167, 172, 173)),
    ("C90", "Charcoal Gray", "#565A5E", (86, 90, 94)),
    ("C91", "Dandelion", "#CEA433", (206, 164, 51)),
    ("C92", "Pale Skin", "#DCB794", (220, 183, 148)),
    ("C93", "Warm Blush", "#DD9285", (221, 146, 133)),
    ("C94", "Salmon", "#E07B69", (224, 123, 105)),
    ("C95", "Apricot", "#EF7F61", (239, 127, 97)),
    ("C96", "Papaya", "#DC772B", (220, 119, 43)),
    ("C97", "Himalaya Blue", "#6AAEDB", (106, 174, 219)),
    ("C98", "Waterfall", "#61BBD3", (97, 187, 211)),
    ("C99", "Lagoon", "#279BBE", (39, 155, 190)),
    ("C100", "Electric Blue", "#00A7E3", (0, 167, 227)),
    ("C101", "Pool Blue", "#0077CA", (0, 119, 202)),
    ("C102", "Caribbean Blue", "#005AA9", (0, 90, 169)),
    ("C103", "Deep Water", "#007F9E", (0, 127, 158)),
    ("C104", "Petrol Blue", "#007D91", (0, 125, 145)),
    ("C105", "Wedgewood Blue", "#00649A", (0, 100, 154)),
    ("C106", "Pond Blue", "#006C9F", (0, 108, 159)),
    ("C107", "Seashell Beige", "#CFC179", (207, 193, 121)),
    ("C108", "Beige", "#C4AE64", (196, 174, 100)),
    ("C109", "Beach Beige", "#AB9745", (171, 151, 69)),
    ("C110", "Caffe Latte", "#978138", (151, 129, 56)),
    ("C111", "Oaktree Brown", "#907C41", (144, 124, 65)),
    ("C112", "Khaki", "#B6AE84", (182, 174, 132)),
    ("C113", "Light Greengray", "#A59F65", (165, 159, 101)),
    ("C114", "Mossy Green", "#938D54", (147, 141, 84)),
    ("C115", "Earth Green", "#8D8B51", (141, 139, 81)),
    ("C116", "Sage Green", "#7F7E49", (127, 126, 73)),
    ("C117", "Pinetree Green", "#5B6E35", (91, 110, 53)),
    ("C118", "Frosty Blue", "#8AD5C9", (138, 213, 201)),
    ("C119", "Polar Mint", "#7CD2A5", (124, 210, 165)),
    ("C120", "Celadon Green", "#72AC9A", (114, 172, 154)),
    ("C121", "Eucalyptus", "#00B26F", (0, 178, 111)),
    ("C122", "Clover Field", "#3EB724", (62, 183, 36)),
    ("C123", "Pooltable Felt", "#0D7535", (13, 117, 53)),
    ("C124", "Snake Green", "#007D6E", (0, 125, 110)),
    ("C125", "Dark Eucalyptus", "#006E69", (0, 110, 105)),
    ("C126", "Marshmallow Rose", "#DFC3E1", (223, 195, 225)),
    ("C127", "Light Grape", "#D38ED4", (211, 142, 212)),
    ("C128", "Rosebud Pink", "#D5A6BA", (213, 166, 186)),
    ("C129", "Fuchsia", "#D6668E", (214, 102, 142)),
    ("C130", "Candy Violet", "#B8AAD9", (184, 170, 217)),
    ("C131", "Flamingo", "#DF486D", (223, 72, 109)),
    ("C132", "Pink Plum", "#BC3CA6", (188, 60, 166)),
    ("C133", "Amethyst", "#803897", (128, 56, 151)),
    ("C134", "Moonlight Blue", "#A7BAE1", (167, 186, 225)),
    ("C135", "Summer Rain", "#AFB8DF", (175, 184, 223)),
    ("C136", "Azur Blue", "#6B9AD4", (107, 154, 212)),
    ("C137", "Cornflower Blue", "#5A89CE", (90, 137, 206)),
    ("C138", "Forget Me Not", "#658AD0", (101, 138, 208)),
    ("C139", "Indigo", "#566CBD", (86, 108, 189)),
    ("C140", "Horizon Blue", "#4D74C6", (77, 116, 198)),
    ("C141", "Cobalt", "#416DBE", (65, 109, 190)),
    ("C142", "Royal Blue", "#30429E", (48, 66, 158)),
    ("C143", "Marine", "#024288", (2, 66, 136)),
    ("C144", "Pale Yellow Moss", "#D6CA6A", (214, 202, 106)),
    ("C145", "Bloodrose Red", "#9D1A38", (157, 26, 56)),
    ("C146", "Spearmint", "#80B7A1", (128, 183, 161)),
    ("C147", "Mocha", "#7A594F", (122, 89, 79)),
    ("C148", "Creme", "#EFDBA1", (239, 219, 161)),
    ("C149", "Iris Violet", "#8884D0", (136, 132, 208)),
    ("C150", "Forrest Green", "#345621", (52, 86, 33)),
    ("C151", "Lilac", "#AEADDC", (174, 173, 220)),
    ("C152", "Pale Lilac", "#BCC3E1", (188, 195, 225)),
    ("C153", "Sahara Sand", "#E3C09A", (227, 192, 154)),
    ("C154", "Sunkissed Teint", "#C58B60", (197, 139, 96)),
    ("C155", "Steel Grey", "#5A5F65", (90, 95, 101)),
    ("C156", "Iron Grey", "#4C5156", (76, 81, 86)),
    ("C157", "Pepper", "#3A3E42", (58, 62, 66)),
]

BEAD_RGB = np.array([c[3] for c in BEAD_COLORS], dtype=np.float64)


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
            code, name, hex_color, _ = BEAD_COLORS[idx]
            row.append({"color": hex_color, "name": name, "code": code})
            color_counts[code] = color_counts.get(code, 0) + 1
        pattern.append(row)

    bead_by_code = {c[0]: c for c in BEAD_COLORS}
    color_summary = [{"code": k, "name": bead_by_code[k][1], "count": v, "hex": bead_by_code[k][2]}
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
    result = image_to_pattern(image_bytes, width, height)
    return jsonify(result)


@app.route("/api/colors")
def colors():
    return jsonify([{"code": c[0], "name": c[1], "hex": c[2]} for c in BEAD_COLORS])


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8892, debug=False)
