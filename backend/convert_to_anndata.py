import pandas as pd
import anndata as ad
import numpy as np

# Load full cell data + ROI flags
df_cells = pd.read_csv("backend/output/cell_features.csv")
df_roi = pd.read_csv("backend/output/roi_cells.csv")

# Merge to get ROI assignment per cell (NaN → not in ROI)
df_cells["roi_score"] = df_cells["cell_id"].map(
    df_roi.set_index("cell_id")["score"]
)
df_cells["roi_id"] = df_cells["cell_id"].map(
    df_roi.set_index("cell_id")["roi_id"]
)
df_cells["roi_score"] = df_cells["roi_score"].fillna(0)
df_cells["roi_id"] = df_cells["roi_id"].fillna("None")

# Features
X = df_cells[["CD31", "CD20", "CD11b", "CD4", "CD11c", "Catalase"]].values

# Cell metadata (obs)
obs = df_cells[["roi_score", "roi_id"]].copy()
obs.index = df_cells["cell_id"].astype(str)
obs.index.name = None  # Remove index name to avoid conflict

# 3D spatial coordinates
obsm = {
    "spatial": df_cells[["x", "y", "z"]].values
}

# Create AnnData object
adata = ad.AnnData(X=X, obs=obs, obsm=obsm)

# Save as Zarr
adata.write_zarr("backend/output/cells.zarr", chunks=(1000,))
print("✅ Saved AnnData to cells.zarr") 