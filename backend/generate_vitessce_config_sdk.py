from vitessce import VitessceConfig
from pathlib import Path
import json

# Initialize Vitessce config
vc = VitessceConfig(name="SpaFGAN_ROI_Viewer", schema_version="1.0.16")
dataset = vc.add_dataset(name="Melanoma", uid="ds")

# Add image.ome-zarr
dataset.add_file(
    file_type="image.ome-zarr",
    url="backend/input/selected_channels.zarr"
)

# Add AnnData cells.zarr
dataset.add_file(
    file_type="anndata.zarr",
    url="backend/output/cells.zarr",
    options={
        "obsIndex": "cell_id",
        "obsEmbedding": {
            "spatial": ["x", "y", "z"]
        },
        "obsFeatureMatrix": {
            "path": ""  # Optional but good practice
        },
        "cellSetObservationField": "interactions"
    }
)

# Add ROI shapes as GeoJSON
dataset.add_file(
    file_type="geojson",
    url="backend/output/roi_shapes.json"
)

# Views
spatial = vc.add_view(dataset=dataset, view_type="spatialBeta")
lc = vc.add_view(dataset=dataset, view_type="layerControllerBeta")
cs = vc.add_view(dataset=dataset, view_type="cellSets")

# Coordination values
vc.link_views_by_dict([spatial, lc, cs], {
    "spatialTargetX": 5500,
    "spatialTargetY": 2880,
    "spatialTargetZ": 0,
    "spatialZoom": -1.1,
    "spatialRotationX": 0,
    "spatialRotationY": 0,
    "spatialRotationZ": 0
})

# Layout
vc.layout(spatial | lc / cs)

# Save config
out_path = Path("backend/output/vitessce_config.json")
out_path.parent.mkdir(parents=True, exist_ok=True)  # ✅ Ensure directory exists

with open(out_path, "w") as f:
    json.dump(vc.to_dict(), f, indent=2)

print(f"✅ Saved Vitessce config to {out_path}")
