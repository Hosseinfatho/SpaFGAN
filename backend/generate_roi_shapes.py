import pandas as pd
import json
from pathlib import Path
import alphashape
from shapely.geometry import mapping, Polygon, MultiPoint
import numpy as np

# Load cell-to-ROI assignments (with markers)
df = pd.read_csv("backend/output/roi_cells.csv")

INTERACTIONS = {
    "T-cell entry site": {"CD31": "high", "CD4": "high"},
    "Inflammatory zone": {"CD11b": "high", "CD20": "high"},
    "Oxidative stress niche": {"CD11b": "high", "Catalase": "high"},
    "B-cell infiltration": {"CD20": "high", "CD31": "high"},
}

THRESHOLDS = {
    "CD31": 1000,
    "CD4": 200,
    "CD11b": 300,
    "CD20": 300,
    "Catalase": 400,
}

def detect_interactions(marker_means, roi_id):
    matched = []
    for label, rule in INTERACTIONS.items():
        match = True
        for marker, level in rule.items():
            value = marker_means.get(marker, 0)
            threshold = THRESHOLDS.get(marker, 500)
            if level == "high" and value <= threshold:
                print(f"[{roi_id}] ❌ {label} NOT matched: {marker}={value} <= {threshold}")
                match = False
                break
        if match:
            print(f"[{roi_id}] ✅ MATCHED: {label}")
            matched.append(label)
    return matched

def round_polygon_coordinates(feature):
    coords = feature["geometry"]["coordinates"]
    if feature["geometry"]["type"] == "Polygon":
        feature["geometry"]["coordinates"] = [[[round(x), round(y)] for x, y in coords[0]]]
    return feature

shapes = []

for roi_id, group in df.groupby("roi_id"):
    points = group[["x", "y"]].dropna().values

    # Ensure marker values are floats
    marker_means = {
        marker: float(group[marker].mean()) if marker in group.columns else 0
        for marker in THRESHOLDS
    }

    print(f"[{roi_id}] Marker means: {marker_means}")
    interactions = detect_interactions(marker_means, roi_id)

    if len(points) < 1:
        continue
    elif len(points) < 3:
        x, y = points[0]
        poly = Polygon([(x, y), (x + 1, y), (x + 1, y + 1), (x, y + 1), (x, y)])
    else:
        try:
            poly = alphashape.alphashape(points, alpha=0.1)
            if poly is None or poly.is_empty:
                poly = MultiPoint(points).convex_hull
        except Exception:
            poly = MultiPoint(points).convex_hull

    shape = {
        "type": "Feature",
        "geometry": mapping(poly),
        "properties": {
            "name": roi_id,
            "type": "ROI",
            "score": float(round(group["score"].mean(), 3)),
            "interactions": interactions
        }
    }

    shapes.append(round_polygon_coordinates(shape))

# Save GeoJSON
out_path = Path("backend/output/roi_shapes.json")
with open(out_path, "w") as f:
    json.dump({"type": "FeatureCollection", "features": shapes}, f, indent=2)

print("✅ ROI shapes written to:", out_path)
