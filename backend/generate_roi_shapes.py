import pandas as pd
import json
from pathlib import Path
import alphashape
from shapely.geometry import mapping, Polygon, MultiPoint
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define interaction logic
INTERACTIONS = {
    "T-cell entry site": {"CD31": "high", "CD4": "high"},
    "Inflammatory zone": {"CD11b": "high", "CD20": "high"},
    "Oxidative stress niche": {"CD11b": "high", "Catalase": "high"},
    "B-cell infiltration": {"CD20": "high", "CD31": "high"},
    "Dendritic signal": {"CD11c": "high"}  # ✅ NEW interaction for CD11c
}


# Thresholds based on previous marker visualizations
THRESHOLDS = {
    "CD31": 1000,
    "CD4": 500,
    "CD11b": 700,
    "CD20": 600,
    "Catalase": 800,
    "CD11c": 400
}

def detect_interactions(marker_means, roi_id):
    matches = []
    for label, rule in INTERACTIONS.items():
        match = True
        for marker, level in rule.items():
            value = marker_means.get(marker, 0)
            threshold = THRESHOLDS.get(marker, 500)
            if level == "high" and value <= threshold:
                logger.debug(f"[{roi_id}] ❌ {label} NOT matched: {marker}={value} <= {threshold}")
                match = False
                break
        if match:
            logger.debug(f"[{roi_id}]  MATCHED: {label}")
            matches.append(label)
    return matches

def round_polygon_coordinates(feature):
    coords = feature["geometry"]["coordinates"]
    if feature["geometry"]["type"] == "Polygon":
        feature["geometry"]["coordinates"] = [[[round(x), round(y)] for x, y in coords[0]]]
    return feature

def process_roi_csv(csv_path):
    try:
        df = pd.read_csv(csv_path)
        if df.empty or df.columns.size == 0:
            logger.warning(f" Skipping empty or invalid file: {csv_path.name}")
            return []
    except pd.errors.EmptyDataError:
        logger.warning(f" Skipping corrupt file: {csv_path.name}")
        return []

    shapes = []

    for roi_id, group in df.groupby("roi_id"):
        points = group[["x", "y"]].dropna().values

        marker_means = {
            marker: float(group[marker].mean()) if marker in group.columns else 0
            for marker in THRESHOLDS
        }

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

    return shapes

def main():
    output_dir = Path(__file__).parent.resolve() / "output"
    csv_files = list(output_dir.glob("roi_cells_*.csv"))
    if not csv_files:
        logger.error(" No ROI cell CSV files found.")
        return

    all_shapes = []
    for csv_file in csv_files:
        logger.info(f" Processing {csv_file.name}")
        shapes = process_roi_csv(csv_file)
        if shapes:
            all_shapes.extend(shapes)

    if not all_shapes:
        logger.warning(" No valid ROI shapes generated.")
        return

    out_path = output_dir / "roi_shapes.json"
    with open(out_path, "w") as f:
        json.dump({"type": "FeatureCollection", "features": all_shapes}, f, indent=2)

    logger.info(f" ROI shapes written to: {out_path}")

if __name__ == "__main__":
    main()
