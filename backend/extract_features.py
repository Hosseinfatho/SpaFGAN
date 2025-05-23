import zarr
import numpy as np
import pandas as pd
from scipy.ndimage import center_of_mass
from pathlib import Path
import logging

# Setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def extract_cell_features(image_zarr_path, segmentation_zarr_path, output_csv_path):
    """Extract features from segmented cells"""
    try:
        # Load image (6, Z, Y, X)
        logger.info("Loading image data...")
        store = zarr.DirectoryStore(str(image_zarr_path))
        z = zarr.open(store, mode="r")
        img = z["data"][:][0]  # remove batch dim
        logger.info(f"Image shape: {img.shape}")

        # Load labels (Z, Y, X)
        logger.info("Loading segmentation mask...")
        store = zarr.DirectoryStore(str(segmentation_zarr_path))
        z = zarr.open(store, mode="r")
        labels = z["segmentation"][:]
        logger.info(f"Labels shape: {labels.shape}")

        # Get unique cell IDs (excluding background)
        cell_ids = np.unique(labels)
        cell_ids = cell_ids[cell_ids != 0]  # remove background
        logger.info(f"Found {len(cell_ids)} segmented cells")

        # Initialize features list
        features = []
        channel_names = ["CD31", "CD20", "CD11b", "CD4", "CD11c", "Catalase"]

        # Process each cell
        for cid in cell_ids:
            logger.info(f"Processing cell {cid}")
            mask = labels == cid

            # Get centroid
            centroid = center_of_mass(mask)
            zc, yc, xc = centroid

            # Mean intensity per channel
            intensity_means = [img[ch][mask].mean() for ch in range(img.shape[0])]

            # Append row: [ID, Z, Y, X, CD31, CD20, CD11b, CD4, CD11c, Catalase]
            features.append([cid, zc, yc, xc] + intensity_means)

        # Create DataFrame
        cols = ["cell_id", "z", "y", "x"] + channel_names
        df = pd.DataFrame(features, columns=cols)

        # Save to CSV
        output_csv_path = Path(output_csv_path)
        output_csv_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_csv_path, index=False)
        logger.info(f"Saved features to {output_csv_path}")

        # Print summary statistics
        logger.info("\nFeature Summary:")
        logger.info(f"Total cells: {len(df)}")
        logger.info("\nCentroid Statistics:")
        logger.info(df[["z", "y", "x"]].describe())
        logger.info("\nIntensity Statistics:")
        logger.info(df[channel_names].describe())

        return df

    except Exception as e:
        logger.error(f"Error extracting features: {e}", exc_info=True)
        return None

def main():
    # Get the absolute path to the backend directory
    backend_dir = Path(__file__).parent.resolve()
    
    # Define paths
    image_zarr_path = backend_dir / "input" / "selected_channels.zarr"
    segmentation_zarr_path = backend_dir / "input" / "segmentation.zarr"
    output_csv_path = backend_dir / "output" / "cell_features.csv"
    
    logger.info(f"Image Zarr path: {image_zarr_path}")
    logger.info(f"Segmentation Zarr path: {segmentation_zarr_path}")
    logger.info(f"Output CSV path: {output_csv_path}")
    
    # Extract features
    df = extract_cell_features(
        image_zarr_path=image_zarr_path,
        segmentation_zarr_path=segmentation_zarr_path,
        output_csv_path=output_csv_path
    )
    
    if df is not None:
        logger.info("✅ Feature extraction completed successfully")
    else:
        logger.error("❌ Feature extraction failed")

if __name__ == "__main__":
    logger.info("Starting feature extraction...")
    main() 