import zarr
import numpy as np
import logging
from pathlib import Path
from skimage.filters import threshold_otsu
from scipy.ndimage import label
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CHANNEL_MAP = {
    "CD31": 0,
    "CD11b": 2,
    "CD11c": 4
}

def load_zarr_data(zarr_path):
    zarr_path = Path(zarr_path).resolve()
    if not zarr_path.exists():
        raise FileNotFoundError(f"Zarr file not found: {zarr_path}")
    store = zarr.DirectoryStore(str(zarr_path))
    z = zarr.open(store, mode="r")
    img = z["data"][:][0]  # Shape: (6, Z, Y, X)
    logger.info(f"Loaded image shape: {img.shape}")
    return img

def segment_cells_3d(image, min_size=50):
    threshold = threshold_otsu(image)
    binary = image > threshold
    labeled, _ = label(binary)
    sizes = np.bincount(labeled.ravel())
    mask_sizes = sizes >= min_size
    mask_sizes[0] = 0
    cleaned = mask_sizes[labeled]
    labeled_cleaned, num = label(cleaned)
    return labeled_cleaned

def save_segmentation(labels, output_path):
    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    store = zarr.DirectoryStore(str(output_path))
    z = zarr.open(store, mode='w')
    z.create_dataset(
        'segmentation',
        data=labels,
        chunks=(1, 688, 1024),
        compressor=zarr.Blosc(cname='zstd', clevel=3)
    )
    z.attrs.update({
        'num_cells': int(labels.max()),
        'shape': labels.shape,
        'dtype': str(labels.dtype),
        'compression': 'zstd',
        'compression_level': 3
    })

def main():
    backend_dir = Path(__file__).parent.resolve()
    zarr_path = backend_dir / "input" / "selected_channels.zarr"
    img = load_zarr_data(zarr_path)

    for marker, index in CHANNEL_MAP.items():
        logger.info(f"\n🔍 Segmenting marker: {marker}")
        labels = segment_cells_3d(img[index], min_size=50)
        output_path = backend_dir / "input" / f"segmentation_{marker}.zarr"
        save_segmentation(labels, output_path)
        logger.info(f"✅ Saved segmentation for {marker} at {output_path}")

if __name__ == "__main__":
    logger.info("🚀 Starting segmentation for all markers...")
    main()
