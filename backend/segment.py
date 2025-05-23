import zarr
import numpy as np
import logging
from pathlib import Path
from skimage.filters import threshold_otsu
from scipy.ndimage import label
from scipy import ndimage
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_zarr_data(zarr_path):
    """Load data from Zarr file"""
    try:
        # Convert to absolute path and validate
        zarr_path = Path(zarr_path).resolve()
        if not zarr_path.exists():
            raise FileNotFoundError(f"Zarr file not found at: {zarr_path}")
            
        logger.info(f"Loading data from {zarr_path}")
        store = zarr.DirectoryStore(str(zarr_path))
        z = zarr.open(store, mode="r")
        
        if "data" not in z:
            raise KeyError("'data' dataset not found in Zarr file")
            
        img = z["data"][:]  # Shape: (1, 6, Z, Y, X)
        img = img[0]        # Remove leading batch dim → (6, Z, Y, X)
        logger.info(f"Loaded image shape: {img.shape}")
        return img
    except Exception as e:
        logger.error(f"Error loading Zarr data: {e}", exc_info=True)
        return None

def segment_cells_3d(image, min_size=50):
    """
    Basic 3D segmentation using Otsu threshold + connected components
    Args:
        image: 3D image array (Z, Y, X)
        min_size: Minimum size of objects to keep
    Returns:
        labeled_cleaned: 3D labeled array
    """
    try:
        logger.info("Starting 3D segmentation...")
        
        # Apply Otsu thresholding
        threshold = threshold_otsu(image)
        binary = image > threshold
        
        # Label connected components
        labeled, num_features = label(binary)
        logger.info(f"Initial number of components: {num_features}")
        
        # Filter small components
        sizes = np.bincount(labeled.ravel())
        mask_sizes = sizes >= min_size
        mask_sizes[0] = 0  # Remove background
        cleaned = mask_sizes[labeled]
        
        # Relabel after cleaning
        labeled_cleaned, num_features = label(cleaned)
        logger.info(f"Final number of cells: {num_features}")
        
        return labeled_cleaned
        
    except Exception as e:
        logger.error(f"Error in 3D segmentation: {e}", exc_info=True)
        return None

def save_segmentation(labels, output_path):
    """Save segmentation results"""
    try:
        # Convert to absolute path and create directory
        output_path = Path(output_path).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Saving segmentation to {output_path}")
        
        # Create Zarr store with compression
        store = zarr.DirectoryStore(str(output_path))
        z = zarr.open(store, mode='w')
        
        # Create dataset with compression settings
        z.create_dataset(
            'segmentation',
            data=labels,
            chunks=(1, 688, 1024),
            compressor=zarr.Blosc(cname='zstd', clevel=3)
        )
        
        # Add metadata
        z.attrs['num_cells'] = labels.max()
        z.attrs['shape'] = labels.shape
        z.attrs['dtype'] = str(labels.dtype)
        z.attrs['compression'] = 'zstd'
        z.attrs['compression_level'] = 3
        
        logger.info(f"Successfully saved segmentation to {output_path}")
        return True
    except Exception as e:
        logger.error(f"Error saving segmentation: {e}", exc_info=True)
        return False

def main():
    # Get the absolute path to the backend directory
    backend_dir = Path(__file__).parent.resolve()
    
    # Define paths relative to backend directory
    zarr_path = backend_dir / "input" / "selected_channels.zarr"
    output_path = backend_dir / "input" / "segmentation.zarr"
    
    logger.info(f"Input Zarr path: {zarr_path}")
    logger.info(f"Output Zarr path: {output_path}")
    
    # Load data
    img = load_zarr_data(zarr_path)
    if img is None:
        return
    
    # Get CD31 channel (index 0)
    cd31_channel = img[0]
    logger.info(f"CD31 channel shape: {cd31_channel.shape}")
    
    # Segment cells using CD31
    labels = segment_cells_3d(cd31_channel, min_size=50)
    if labels is None:
        return
    
    # Save results
    save_segmentation(labels, output_path)

if __name__ == "__main__":
    logger.info("Starting cell segmentation...")
    main()
    logger.info("✅ Segmentation completed") 