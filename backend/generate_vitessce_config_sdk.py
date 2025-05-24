from vitessce import VitessceConfig
from pathlib import Path
import json

vc = VitessceConfig(name="SpaFGAN_ROI_Viewer", schema_version="1.0.16")
dataset = vc.add_dataset(name="Melanoma", uid="ds")

# Add OME-Zarr image
dataset.add_file(
    file_type="image.ome-zarr",
    url="backend/input/selected_channels.zarr"
)

# Add AnnData (cells.zarr)
dataset.add_file(
    file_type="anndata.zarr",
    url="backend/output/cells.zarr",
    options={
        "obsIndex": "cell_id",
        "obsEmbedding": {"spatial": ["x", "y", "z"]}
    }
)

# Add ROI shapes
dataset.add_file(
    file_type="raster.json",
    url="backend/output/roi_shapes.json"
)

# Add views
spatial = vc.add_view(dataset=dataset, view_type="spatialBeta")
lc = vc.add_view(dataset=dataset, view_type="layerControllerBeta")

vc.link_views_by_dict([spatial, lc], {
    "spatialTargetX": 4000,
    "spatialTargetY": 3000,
    "spatialTargetZ": 25,
    "spatialZoom": -1.2
})

vc.layout(spatial | lc)

# Save config
out_path = Path("backend/output/vitessce_config.json")
with open(out_path, "w") as f:
    json.dump(vc.to_dict(), f, indent=2)
print(f"✅ Saved Vitessce config to {out_path}") 