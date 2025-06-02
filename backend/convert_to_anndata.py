import pandas as pd
import anndata as ad
import numpy as np
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Thresholds for interaction detection
THRESHOLDS = {
    "CD31": 1000,
    "CD20": 600,
    "CD11b": 700,
    "CD4": 500,
    "Catalase": 800,
    "CD11c": 400,
}

INTERACTIONS = {
    "T-cell entry site": {"CD31": "high", "CD4": "high"},
    "Inflammatory zone": {"CD11b": "high", "CD20": "high"},
    "Oxidative stress niche": {"CD11b": "high", "Catalase": "high"},
    "B-cell infiltration": {"CD20": "high", "CD31": "high"},
    "Dendritic signal": {"CD11c": "high"}
}

def detect_interactions(row):
    found = []
    for label, rule in INTERACTIONS.items():
        if all(row.get(m, 0) > THRESHOLDS.get(m, 0) for m in rule):
            found.append(label)
    return ", ".join(found) if found else "None"

def main():
    input_dir = Path(__file__).parent / "output"

    # Load all cell features
    feature_files = sorted(input_dir.glob("cell_features_*.csv"))
    if not feature_files:
        logger.error(" No cell_features_*.csv files found.")
        return

    df_cells = pd.concat([pd.read_csv(f) for f in feature_files], ignore_index=True)
    logger.info(f"Loaded {len(df_cells)} total cells")

    # Load all non-empty ROI files
    roi_feature_files = sorted(input_dir.glob("roi_cells_*.csv"))
    valid_roi_files = []
    for f in roi_feature_files:
        try:
            if f.stat().st_size > 0:
                df = pd.read_csv(f)
                if not df.empty and "cell_id" in df.columns:
                    valid_roi_files.append(df)
        except Exception as e:
            logger.warning(f"⚠️ Skipping corrupt or unreadable file: {f.name}")

    if not valid_roi_files:
        logger.warning(" No valid ROI cell files found. Proceeding without ROI annotations.")
        df_cells["roi_score"] = 0
        df_cells["roi_id"] = "None"
    else:
        df_roi = pd.concat(valid_roi_files, ignore_index=True)
        
        # Aggregate scores and ROI IDs per cell (in case of duplicates)
        df_roi_grouped = df_roi.groupby("cell_id").agg({
            "spafgan_score": "max",  # Use max score per cell_id
            "roi_id": "first"       # Or use any consistent rule like 'first'
        })
         
        duplicates = df_roi[df_roi.duplicated("cell_id", keep=False)]
        if not duplicates.empty:
            print(" Duplicate cell_id rows:\n", duplicates)

        # Map to main cell table
        df_cells["roi_score"] = df_cells["cell_id"].map(df_roi_grouped["spafgan_score"])
        df_cells["roi_id"] = df_cells["cell_id"].map(df_roi_grouped["roi_id"])

        df_cells["roi_score"] = df_cells["roi_score"].fillna(0)
        df_cells["roi_id"] = df_cells["roi_id"].fillna("None")

    # Compute interaction labels
    df_cells["interactions"] = df_cells.apply(detect_interactions, axis=1)

    # Create AnnData
    X = df_cells[["CD31", "CD20", "CD11b", "CD4", "CD11c", "Catalase"]].values
    obs = df_cells[["roi_score", "roi_id", "interactions"]].copy()
    obs.index = df_cells["cell_id"].astype(str)
    obsm = {"spatial": df_cells[["x", "y", "z"]].values}

    adata = ad.AnnData(X=X, obs=obs, obsm=obsm)
    adata.write_zarr(input_dir / "cells.zarr", chunks=(1000,))

    logger.info("✅ Saved AnnData to cells.zarr with interaction labels")

if __name__ == "__main__":
    main()
