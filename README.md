# WasteVision — YOLO Waste Analyzer

A web app that uses your `best.pt` YOLOv6s model to analyze images of waste, providing counts, category breakdowns, and estimated weight.

## Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Place your best.pt in the same folder as app.py
#    (or set MODEL_PATH env variable to its location)
export MODEL_PATH=/path/to/best.pt   # optional, default = ./best.pt

# 3. Run the server
python app.py
```

Open http://localhost:5000 in your browser.

## Usage

1. Enter the **full path** to a folder containing images (JPG, PNG, BMP, WebP, TIFF).
2. Click **▶ Analyze**.
3. View the dashboard:
   - **Overview** — total images, detections, estimated weight
   - **Detected Items** — per-class counts & weight
   - **Weight Distribution** — visual bar
   - **Waste Categories** — Human/Consumer, Industrial, Hazardous
   - **Per-Image Breakdown** — table of every image

## Weight Estimates

| Class   | Weight per item |
|---------|----------------|
| Bottle  | 25 g (500ml PET bottle) |
| Carton  | 180 g (avg food/beverage carton) |

## Category Mapping

| Detected Class | Category | Subcategory |
|---------------|----------|-------------|
| bottle | Human / Consumer Waste | Single-use Plastics |
| carton | Human / Consumer Waste | Food & Beverage Packaging |

Industrial and Hazardous categories are shown as empty placeholders — extend `CATEGORIES` and `WEIGHT_G` in `app.py` to add more classes.
