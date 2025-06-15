import zarr
import numpy as np
import pandas as pd
from scipy.ndimage import center_of_mass
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Channel names in the same order as selected_channels.zarr
channel_names = ["CD31", "CD20", "CD11b", "CD4", "CD11c", "Catalase"]

# Define markers for separate segmentation and output
MARKERS = {
    "CD31": "segmentation_CD31.zarr",
    "CD11b": "segmentation_CD11b.zarr",
    "CD11c": "segmentation_CD11c.zarr",
    "CD4": "segmentation_CD4.zarr",
    "CD20": "segmentation_CD20.zarr",
    "Catalase": "segmentation_Catalase.zarr"
}

def extract_cell_features(image, labels, marker_name):
    """Extract cell features for one marker"""
    cell_ids = np.unique(labels)
    cell_ids = cell_ids[cell_ids != 0]  # remove background
    logger.info(f"[{marker_name}] Found {len(cell_ids)} segmented cells")

    features = []
    for cid in cell_ids:
        mask = labels == cid
        # Log mask size
        mask_size = np.sum(mask)
        logger.info(f"Cell {cid} mask size: {mask_size} voxels")
        
        # Calculate centroid and round to nearest integer
        zc, yc, xc = [int(round(coord)) for coord in center_of_mass(mask)]
        # Calculate mean intensities and round to 2 decimal places
        intensity_means = [round(image[ch][mask].mean(), 2) for ch in range(image.shape[0])]
        features.append([int(cid), zc, yc, xc] + intensity_means)

    cols = ["cell_id", "z", "y", "x"] + channel_names
    df = pd.DataFrame(features, columns=cols)
    
    # Convert all columns to appropriate types
    df['cell_id'] = df['cell_id'].astype(int)
    df[['z', 'y', 'x']] = df[['z', 'y', 'x']].astype(int)
    df[channel_names] = df[channel_names].round(2)
    
    return df

def main():
    backend_dir = Path(__file__).parent.resolve()
    input_dir = backend_dir / "input"
    output_dir = backend_dir / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    image_zarr_path = input_dir / "selected_channels.zarr"
    logger.info(f"Loading image from {image_zarr_path}")
    store = zarr.DirectoryStore(str(image_zarr_path))
    img = zarr.open(store, mode="r")["data"][:][0]  # (6, Z, Y, X)
    logger.info(f"Image shape: {img.shape}")

    for marker, seg_file in MARKERS.items():
        seg_path = input_dir / seg_file
        if not seg_path.exists():
            logger.warning(f"Segmentation file for {marker} not found: {seg_path}")
            continue

        logger.info(f"Loading segmentation for {marker} from {seg_path}")
        labels = zarr.open(zarr.DirectoryStore(str(seg_path)), mode="r")["segmentation"][:]

        df = extract_cell_features(img, labels, marker)
        output_csv_path = output_dir / f"cell_features_{marker}.csv"
        df.to_csv(output_csv_path, index=False)
        logger.info(f"[{marker}] Features saved to {output_csv_path}")

        # Print summary statistics
        logger.info(f"\n[{marker}] Centroid Stats:\n{df[['z', 'y', 'x']].describe()}")
        logger.info(f"\n[{marker}] Intensity Stats:\n{df[channel_names].describe()}")

if __name__ == "__main__":
    logger.info(" Starting multi-marker feature extraction...")
    main()
    logger.info(" Done.")
