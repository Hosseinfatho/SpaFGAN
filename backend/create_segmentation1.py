import zarr
import numpy as np
import logging
from pathlib import Path
import dask.array as da
from dask.diagnostics import ProgressBar
from scipy.ndimage import label, gaussian_filter
from skimage.filters import threshold_otsu

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define markers for segmentation with their thresholds
MARKERS = {
    "CD31": {"index": 0, "threshold": 0.1},    # Blood vessel cells
    "CD11b": {"index": 2, "threshold": 0.08},   # Immune cells
    "CD11c": {"index": 4, "threshold": 0.08},   # Dendritic cells
    "CD4": {"index": 1, "threshold": 0.08},     # T cells
    "CD20": {"index": 3, "threshold": 0.1},    # B cells
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

def create_segmentation_optimized(data, marker_name, marker_info):
    """Create segmentation for a specific marker - optimized version"""
    try:
        # Get marker data - process in chunks
        marker_data = data[0, marker_info["index"]].compute()
        
        logger.info(f"Processing {marker_name} - data shape: {marker_data.shape}")
        
        # Normalize data efficiently
        min_val = np.min(marker_data)
        max_val = np.max(marker_data)
        normalized_data = (marker_data - min_val) / (max_val - min_val)
        
        logger.info(f"{marker_name} - Min: {min_val:.3f}, Max: {max_val:.3f}")
        
        # Calculate Otsu's threshold
        otsu_threshold = threshold_otsu(normalized_data)
        logger.info(f"{marker_name} - Otsu threshold: {otsu_threshold:.3f}, Manual threshold: {marker_info['threshold']:.3f}")
        
        # Apply Gaussian blur to reduce noise
        smoothed = gaussian_filter(normalized_data, sigma=1.0)
        
        # Use manual threshold only (no adaptive thresholding)
        manual_threshold = marker_info["threshold"]
        adaptive_threshold = manual_threshold  # Use only manual threshold
        
        logger.info(f"{marker_name} - Manual threshold: {adaptive_threshold:.3f}")
        
        # Apply thresholding
        binary = smoothed > adaptive_threshold
        
        # Apply morphological operations efficiently - lighter operations
        from scipy.ndimage import binary_opening, binary_closing
        
        # Remove small noise with opening - use smaller 3D structure
        binary = binary_opening(binary, structure=np.ones((2, 2, 2)))
        
        # Fill small holes with closing - use smaller 3D structure
        binary = binary_closing(binary, structure=np.ones((2, 2, 2)))
        
        # Label connected components
        labeled_array, num_areas = label(binary)
        
        # Filter out areas with less than 5 voxels
        min_area_size = 5
        filtered_labels = np.zeros_like(labeled_array)
        valid_areas = 0
        
        logger.info(f"Found {num_areas} initial areas for {marker_name}")
        
        for i in range(1, num_areas + 1):
            area_mask = (labeled_array == i)
            area_size = np.sum(area_mask)
            
            if area_size >= min_area_size:
                filtered_labels[area_mask] = valid_areas + 1
                valid_areas += 1
                logger.info(f"Area {i}: {area_size} voxels - KEPT")
            else:
                logger.info(f"Area {i}: {area_size} voxels - REMOVED (too small)")
        
        logger.info(f"Created {marker_name} segmentation with {valid_areas} valid areas (filtered from {num_areas} total areas)")
        
        # Calculate area statistics
        if valid_areas > 0:
            area_sizes = []
            for i in range(1, valid_areas + 1):
                area_size = np.sum(filtered_labels == i)
                area_sizes.append(area_size)
            
            avg_area_size = np.mean(area_sizes)
            max_area_size = np.max(area_sizes)
            min_area_size_found = np.min(area_sizes)
            
            logger.info(f"{marker_name} area statistics:")
            logger.info(f"  - Average area size: {avg_area_size:.1f} voxels")
            logger.info(f"  - Largest area: {max_area_size} voxels")
            logger.info(f"  - Smallest area: {min_area_size_found} voxels")
        
        return filtered_labels
        
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
    """Create segmentation files for all markers - optimized version"""
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
            segmentation = create_segmentation_optimized(data, marker_name, marker_info)
            if segmentation is not None:
                # Save segmentation
                seg_path = input_dir / f"segmentation_{marker_name}.zarr"
                save_segmentation(segmentation, seg_path, data.chunksize[2:])

        return True

    except Exception as e:
        logger.error(f"Error in create_all_segmentations: {e}", exc_info=True)
        return False

if __name__ == "__main__":
    create_all_segmentations() 