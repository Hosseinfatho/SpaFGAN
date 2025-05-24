import pandas as pd
import anndata as ad
import numpy as np

# Define marker thresholds for detecting high expression
THRESHOLDS = {
    "CD31": 1000,
    "CD20": 600,
    "CD11b": 700,
    "CD4": 500,
    "Catalase": 800,
}

# Define cell-cell interaction logic
INTERACTIONS = {
    "T-cell entry site": {"CD31": "high", "CD4": "high"},
    "Inflammatory zone": {"CD11b": "high", "CD20": "high"},
    "Oxidative stress niche": {"CD11b": "high", "Catalase": "high"},
    "B-cell infiltration": {"CD20": "high", "CD31": "high"},
}

def detect_interactions(row):
    found = []
    for label, rule in INTERACTIONS.items():
        match = True
        for marker, level in rule.items():
            if level == "high" and row[marker] <= THRESHOLDS[marker]:
                match = False
                break
        if match:
            found.append(label)
    return ", ".join(found) if found else "None"

# Load full cell data + ROI flags
df_cells = pd.read_csv("backend/output/cell_features.csv")
df_roi = pd.read_csv("backend/output/roi_cells.csv")

# Merge to get ROI assignment per cell (NaN → not in ROI)
df_cells["roi_score"] = df_cells["cell_id"].map(df_roi.set_index("cell_id")["score"])
df_cells["roi_id"] = df_cells["cell_id"].map(df_roi.set_index("cell_id")["roi_id"])
df_cells["roi_score"] = df_cells["roi_score"].fillna(0)
df_cells["roi_id"] = df_cells["roi_id"].fillna("None")

# Detect interaction types
df_cells["interactions"] = df_cells.apply(detect_interactions, axis=1)

# Features matrix
X = df_cells[["CD31", "CD20", "CD11b", "CD4", "CD11c", "Catalase"]].values

# Cell metadata (obs)
obs = df_cells[["roi_score", "roi_id", "interactions"]].copy()
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
print("✅ Saved AnnData to cells.zarr with interaction labels")
