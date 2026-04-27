import os
import json
from pathlib import Path
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

app = Flask(__name__, static_folder="static")
CORS(app)

# ── Weight estimates (grams per item) ──────────────────────────────────────────
WEIGHT_G = {
    "bottle": 25,    # typical 500 ml PET bottle
    "carton": 180,   # average beverage / food carton
}

# ── Waste category mapping ─────────────────────────────────────────────────────
CATEGORIES = {
    "bottle": {
        "category": "Human / Consumer Waste",
        "subcategory": "Single-use Plastics",
        "icon": "🍶",
        "color": "#4FC3F7",
    },
    "carton": {
        "category": "Human / Consumer Waste",
        "subcategory": "Food & Beverage Packaging",
        "icon": "📦",
        "color": "#A5D6A7",
    },
}

# ── Lazily load the YOLO model ─────────────────────────────────────────────────
_model = None

def get_model():
    global _model
    if _model is None:
        from ultralytics import YOLO
        model_path = os.environ.get("MODEL_PATH", "best.onnx")
        _model = YOLO(model_path)
    return _model


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}


@app.route("/")
def index():
    return send_from_directory("static", "static/index.html")


@app.route("/api/process", methods=["POST"])
def process_folder():
    data = request.get_json()
    folder = data.get("folder", "").strip()

    if not folder:
        return jsonify({"error": "No folder path provided."}), 400

    folder_path = Path(folder)
    if not folder_path.exists() or not folder_path.is_dir():
        return jsonify({"error": f"Folder not found: {folder}"}), 400

    images = [p for p in folder_path.iterdir() if p.suffix.lower() in IMAGE_EXTS]
    if not images:
        return jsonify({"error": "No images found in the specified folder."}), 400

    model = get_model()
    counts = {}          # class_name -> count
    per_image = []       # per-image breakdown

    for img_path in sorted(images):
        results = model(str(img_path), verbose=False)
        image_counts = {}

        for result in results:
            if result.boxes is None:
                continue
            for cls_id in result.boxes.cls.tolist():
                cls_name = model.names[int(cls_id)].lower()
                counts[cls_name] = counts.get(cls_name, 0) + 1
                image_counts[cls_name] = image_counts.get(cls_name, 0) + 1

        per_image.append({
            "file": img_path.name,
            "detections": image_counts,
        })

    # ── Build category summary ─────────────────────────────────────────────────
    category_totals = {}
    total_weight_g = 0

    for cls_name, count in counts.items():
        weight_g = WEIGHT_G.get(cls_name, 100) * count
        total_weight_g += weight_g
        cat_info = CATEGORIES.get(cls_name, {
            "category": "Other Waste",
            "subcategory": "Unclassified",
            "icon": "🗑️",
            "color": "#CE93D8",
        })
        cat_key = cat_info["category"]
        if cat_key not in category_totals:
            category_totals[cat_key] = {
                "category": cat_key,
                "icon": cat_info["icon"],
                "color": cat_info["color"],
                "items": [],
                "weight_g": 0,
            }
        category_totals[cat_key]["items"].append({
            "type": cls_name,
            "subcategory": cat_info["subcategory"],
            "count": count,
            "weight_g": weight_g,
            "icon": cat_info["icon"],
        })
        category_totals[cat_key]["weight_g"] += weight_g

    # Add placeholder categories for context (0 detections)
    placeholder_categories = [
        {
            "category": "Industrial Waste",
            "icon": "🏭",
            "color": "#FFAB91",
            "items": [],
            "weight_g": 0,
            "placeholder": True,
        },
        {
            "category": "Hazardous Waste",
            "icon": "☣️",
            "color": "#EF9A9A",
            "items": [],
            "weight_g": 0,
            "placeholder": True,
        },
    ]

    all_categories = list(category_totals.values()) + placeholder_categories

    return jsonify({
        "total_images": len(images),
        "total_detections": sum(counts.values()),
        "counts": counts,
        "total_weight_kg": round(total_weight_g / 1000, 3),
        "categories": all_categories,
        "per_image": per_image,
    })


if __name__ == "__main__":
    app.run(debug=True, port=5000)
