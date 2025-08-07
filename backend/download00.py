import zarr
import numpy as np
import s3fs
import logging
from pathlib import Path
import os
import dask.array as da
from dask.diagnostics import ProgressBar

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# S3 path settings
ZARR_BASE_URL = "s3://lsp-public-data/biomedvis-challenge-2025/Dataset1-LSP13626-melanoma-in-situ"
ZARR_IMAGE_GROUP_PATH = "0"
TARGET_RESOLUTION_PATH = ""  # Using root data directory

# Define channel indices
CHANNEL_INDICES = {
    'CD31': 19,
    'CD20': 27,
    'CD11b': 37,
    'CD4': 25,
    'CD11c': 41,
    'Catalase': 59,
}

def extract_channels_from_zarr():
    """Extract selected channels from Zarr using dask"""
    try:
        # Create S3 filesystem
        logger.info("Creating S3 filesystem...")
        s3 = s3fs.S3FileSystem(anon=True)
        
        # Construct the S3 path
        s3_path = "lsp-public-data/biomedvis-challenge-2025/Dataset1-LSP13626-melanoma-in-situ/0"
        logger.info(f"Accessing S3 path: {s3_path}")
        
        # List available files to understand structure
        files = s3.ls(s3_path)
        logger.info(f"Available files in S3 path: {files}")
        
        # Look for the data array in resolution levels
        data_path = None
        
        # Use resolution level 3 specifically
        resolution = '3'
        resolution_path = f"{s3_path}/{resolution}"
        try:
            resolution_files = s3.ls(resolution_path)
            logger.info(f"Resolution {resolution} files: {resolution_files}")
            
            # Check if this resolution level has a data array
            for file_path in resolution_files:
                if file_path.endswith('/data'):
                    data_path = file_path
                    logger.info(f"Found data array at: {data_path}")
                    break
                elif file_path.endswith('/.zarray'):
                    # This indicates an array at this level
                    data_path = file_path.replace('/.zarray', '')
                    logger.info(f"Found array at: {data_path}")
                    break
            
        except Exception as e:
            logger.error(f"Could not access resolution {resolution}: {e}")
            return None, None, None
        
        if data_path is None:
            logger.error("Could not find data array in any resolution level")
            return None, None, None
        
        logger.info(f"Using data path: {data_path}")
        
        # Load the data using dask with s3fs
        logger.info("Loading data using dask...")
        s3_url = f"s3://{data_path}"
        logger.info(f"Loading from S3 URL: {s3_url}")
        dask_array = da.from_zarr(s3_url, storage_options={'anon': True})
        logger.info(f"Dask array shape: {dask_array.shape}")
        logger.info(f"Chunk size: {dask_array.chunksize}")
        
        # Select channels
        channel_indices = list(CHANNEL_INDICES.values())
        logger.info(f"Selecting channels: {channel_indices}")
        
        # Create a new dask array with selected channels
        selected_channels = dask_array[:, channel_indices, :, :, :]
        logger.info(f"Selected channels shape: {selected_channels.shape}")
        
        return selected_channels, channel_indices, dask_array.chunksize
        
    except Exception as e:
        logger.error(f"Error extracting channels: {e}", exc_info=True)
        return None, None, None

def save_as_zarr(data, output_path, channel_names, chunks):
    """Save data as Zarr format"""
    try:
        # Create output directory
        output_path = Path(output_path).resolve()
        output_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Creating output directory at: {output_path}")

        # Create Zarr store - use the path directly
        store = str(output_path)
        root = zarr.group(store=store, overwrite=True)

        # Add metadata
        root.attrs['channel_names'] = channel_names
        root.attrs['axes'] = ['t', 'c', 'z', 'y', 'x']
        root.attrs['dimensions'] = {
            't': 1,
            'c': len(channel_names),
            'z': data.shape[2],
            'y': data.shape[3],
            'x': data.shape[4]
        }

        # Create dataset
        dataset = root.create_dataset(
            'data',
            shape=data.shape,
            chunks=chunks,
            dtype=data.dtype
        )

        # Write data with progress bar
        with ProgressBar():
            da.store(data, dataset)

        logger.info(f"Data saved as Zarr in {output_path}")
        return True
    except Exception as e:
        logger.error(f"Error saving Zarr: {e}", exc_info=True)
        return False

def download_channels():
    """Download and save selected channels from Zarr data"""
    try:
        # Extract channels using dask
        selected_channels, channel_indices, chunks = extract_channels_from_zarr()
        if selected_channels is None:
            return False

        # Get the absolute path to the backend directory
        backend_dir = Path(__file__).parent.resolve()
        input_dir = backend_dir / "input"
        input_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created input directory at: {input_dir}")

        # Save channels as Zarr
        zarr_path = input_dir / "selected_channels.zarr"
        channel_names = [name for name in CHANNEL_INDICES.keys()]
        save_as_zarr(selected_channels, zarr_path, channel_names, chunks)

        # Save metadata
        metadata = {
            'shape': selected_channels.shape,
            'dtype': str(selected_channels.dtype),
            'channel_indices': channel_indices,
            'channel_names': CHANNEL_INDICES,
            'resolution_level': TARGET_RESOLUTION_PATH,
            'chunks': chunks
        }
        metadata_path = input_dir / "metadata.npy"
        np.save(metadata_path, metadata)
        logger.info(f"Metadata saved at: {metadata_path}")

        return True

    except Exception as e:
        logger.error(f"Error downloading channels: {e}", exc_info=True)
        return False

if __name__ == "__main__":
    logger.info("Starting channel download...")
    success = download_channels()
    if success:
        logger.info(" Channel download completed successfully")
    else:
        logger.error(" Channel download failed") 