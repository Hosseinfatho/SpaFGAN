import zarr
import numpy as np
import logging
from pathlib import Path
import dask.array as da
from dask.diagnostics import ProgressBar
from scipy.ndimage import label, gaussian_filter
from skimage.filters import threshold_otsu
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define markers for segmentation with their thresholds
MARKERS = {
    "CD31": {"index": 0, "threshold": 0.1},    # Blood vessel cells
    "CD11b": {"index": 2, "threshold": 0.1},   # Immune cells
    "CD11c": {"index": 4, "threshold": 0.1},   # Dendritic cells
    "CD4": {"index": 1, "threshold": 0.1},    # T cells
    "CD20": {"index": 3, "threshold": 0.1},   # B cells
    "Catalase": {"index": 5, "threshold": 0.1} # Oxidative stress marker
}

def load_channels():
    """Load selected_channels.zarr"""
    try:
        # Get the absolute path to the backend directory
        backend_dir = Path(__file__).parent.resolve()
        input_dir = backend_dir / "input"
        zarr_path = input_dir / "selected_channels.zarr"
        
        # Load the data
        logger.info(f"Loading data from {zarr_path}")
        store = zarr.DirectoryStore(str(zarr_path))
        root = zarr.group(store=store)
        data = da.from_zarr(root['data'])
        
        logger.info(f"Data shape: {data.shape}")
        return data
        
    except Exception as e:
        logger.error(f"Error loading channels: {e}", exc_info=True)
        return None

def normalize_data(data):
    """Normalize data to range [0, 1]"""
    min_val = np.min(data)
    max_val = np.max(data)
    normalized = (data - min_val) / (max_val - min_val)
    return normalized, min_val, max_val

def analyze_threshold(data, marker_name, marker_index, threshold):
    """Analyze and visualize threshold for a marker"""
    try:
        # Get marker data
        marker_data = data[0, marker_index].compute()
        
        # Normalize data
        normalized_data, min_val, max_val = normalize_data(marker_data)
        logger.info(f"{marker_name} - Min: {min_val:.3f}, Max: {max_val:.3f}")
        
        # Calculate Otsu's threshold on normalized data
        otsu_threshold = threshold_otsu(normalized_data)
        
        # Create histogram
        plt.figure(figsize=(12, 8))
        
        # Plot original data histogram
        plt.subplot(2, 1, 1)
        plt.hist(marker_data.flatten(), bins=100)
        plt.title(f'{marker_name} Original Intensity Distribution')
        plt.xlabel('Original Intensity')
        plt.ylabel('Frequency')
        
        # Plot normalized data histogram
        plt.subplot(2, 1, 2)
        plt.hist(normalized_data.flatten(), bins=100)
        plt.axvline(x=threshold, color='r', linestyle='--', 
                   label=f'Manual threshold: {threshold:.2f}')
        plt.axvline(x=otsu_threshold, color='g', linestyle='--', 
                   label=f'Otsu threshold: {otsu_threshold:.2f}')
        plt.title(f'{marker_name} Normalized Intensity Distribution')
        plt.xlabel('Normalized Intensity [0,1]')
        plt.ylabel('Frequency')
        plt.legend()
        
        # Save plot
        backend_dir = Path(__file__).parent.resolve()
        input_dir = backend_dir / "input"
        plt.savefig(input_dir / f"{marker_name}_threshold_analysis.png")
        plt.close()
        
        return otsu_threshold, normalized_data
        
    except Exception as e:
        logger.error(f"Error analyzing threshold: {e}", exc_info=True)
        return None, None

def create_segmentation(data, marker_name, marker_info):
    """Create segmentation for a specific marker"""
    try:
        # Get marker data
        marker_data = data[0, marker_info["index"]].compute()
        
        # Analyze threshold and get normalized data
        otsu_threshold, normalized_data = analyze_threshold(
            data, marker_name, marker_info["index"], marker_info["threshold"]
        )
        
        logger.info(f"{marker_name} - Otsu threshold: {otsu_threshold:.3f}, "
                   f"Manual threshold: {marker_info['threshold']:.3f}")
        
        # Apply Gaussian blur to reduce noise
        smoothed = gaussian_filter(normalized_data, sigma=1)
        
        # Apply thresholding
        binary = smoothed > marker_info["threshold"]
        
        # Label connected components
        labeled_array, num_features = label(binary)
        
        logger.info(f"Created {marker_name} segmentation with {num_features} cells")
        return labeled_array
        
    except Exception as e:
        logger.error(f"Error creating {marker_name} segmentation: {e}", exc_info=True)
        return None

def save_segmentation(data, output_path, chunks):
    """Save segmentation as Zarr format"""
    try:
        # Create output directory
        output_path = Path(output_path).resolve()
        output_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Creating output directory at: {output_path}")

        # Create Zarr store
        store = zarr.DirectoryStore(str(output_path))
        root = zarr.group(store=store, overwrite=True)

        # Add metadata
        root.attrs['axes'] = ['z', 'y', 'x']
        root.attrs['dimensions'] = {
            'z': data.shape[0],
            'y': data.shape[1],
            'x': data.shape[2]
        }

        # Create dataset
        dataset = root.create_dataset(
            'segmentation',
            shape=data.shape,
            chunks=chunks,
            dtype=data.dtype,
            compressor=zarr.Blosc(cname='zstd', clevel=3)
        )

        # Convert numpy array to dask array
        dask_data = da.from_array(data, chunks=chunks)

        # Write data with progress bar
        with ProgressBar():
            da.store(dask_data, dataset)

        logger.info(f"Segmentation saved as Zarr in {output_path}")
        return True
    except Exception as e:
        logger.error(f"Error saving Zarr: {e}", exc_info=True)
        return False

def create_all_segmentations():
    """Create segmentation files for all markers"""
    try:
        # Load channel data
        data = load_channels()
        if data is None:
            return False

        # Get the absolute path to the backend directory
        backend_dir = Path(__file__).parent.resolve()
        input_dir = backend_dir / "input"
        input_dir.mkdir(parents=True, exist_ok=True)

        # Process each marker
        for marker_name, marker_info in MARKERS.items():
            logger.info(f"\nProcessing {marker_name}...")
            
            # Create segmentation
            segmentation = create_segmentation(data, marker_name, marker_info)
            if segmentation is not None:
                # Save segmentation
                seg_path = input_dir / f"segmentation_{marker_name}.zarr"
                save_segmentation(segmentation, seg_path, data.chunksize[2:])

        return True

    except Exception as e:
        logger.error(f"Error creating segmentations: {e}", exc_info=True)
        return False

if __name__ == "__main__":
    logger.info("Starting segmentation creation...")
    success = create_all_segmentations()
    if success:
        logger.info("Segmentation creation completed successfully")
    else:
        logger.error("Segmentation creation failed") 