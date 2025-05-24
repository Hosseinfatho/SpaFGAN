import pandas as pd
import json
from pathlib import Path
import alphashape
from shapely.geometry import mapping, Polygon, MultiPoint

# Load ROI cell positions
df = pd.read_csv("backend/output/roi_cells.csv")

# Prepare output
shapes = []
for roi_id, group in df.groupby("roi_id"):
    points_2d = group[["x", "y"]].values  # Project to XY for now

    if len(points_2d) < 3:
        # Not enough points for a shape, make a bounding box
        x0, y0 = points_2d[0]
        polygon = Polygon([(x0, y0), (x0+1, y0), (x0+1, y0+1), (x0, y0+1)])
    else:
        shape = alphashape.alphashape(points_2d, alpha=0.1)
        polygon = shape if shape.is_valid else shape.convex_hull

    shape_json = {
        "type": "Feature",
        "geometry": mapping(polygon),
        "properties": {
            "name": roi_id,
            "type": "ROI"
        }
    }
    shapes.append(shape_json)

# Save to JSON
output_path = Path("backend/output/roi_shapes.json")
with open(output_path, "w") as f:
    json.dump({"type": "FeatureCollection", "features": shapes}, f, indent=2)

print(f"✅ Saved ROI shape overlay to {output_path}") 