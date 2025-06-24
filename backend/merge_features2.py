import pandas as pd
import numpy as np
import logging
from pathlib import Path
import zarr
from scipy.ndimage import label, center_of_mass
from scipy.spatial.distance import cdist

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define markers
MARKERS = ["CD31", "CD20", "CD11b", "CD4", "CD11c", "Catalase"]
MARKER_INDICES = {"CD31": 0, "CD20": 1, "CD11b": 2, "CD4": 3, "CD11c": 4, "Catalase": 5}

# Define thresholds for each marker (same as in create_segmentation1.py)
MARKER_THRESHOLDS = {
    "CD31": 0.1,
    "CD20": 0.1,
    "CD11b": 0.08,
    "CD4": 0.08,
    "CD11c": 0.08,
    "Catalase": 0.1
}

def load_original_data():
    """Load original zarr data for intensity values"""
    try:
        backend_dir = Path(__file__).parent.resolve()
        input_dir = backend_dir / "input"
        zarr_path = input_dir / "selected_channels.zarr"
        
        logger.info(f"Loading original data from {zarr_path}")
        store = zarr.DirectoryStore(str(zarr_path))
        root = zarr.group(store=store)
        data = root['data'][:]  # Load all data into memory
        
        logger.info(f"Original data shape: {data.shape}")
        return data
        
    except Exception as e:
        logger.error(f"Error loading original data: {e}")
        return None

def calculate_region_radius(area_size):
    """Calculate radius based on area size"""
    # Radius = cube root of volume
    volume = area_size
    radius = int(np.cbrt(volume))
    return max(radius, 3)  # Minimum radius of 3

def get_marker_intensities_in_region_with_threshold(center, radius, original_data):
    """Calculate intensities of all markers in a region with threshold filtering"""
    z, y, x = center
    
    # Define region bounds - ensure all are integers
    z_min = int(max(0, z-radius))
    z_max = int(min(original_data.shape[2], z+radius))
    y_min = int(max(0, y-radius))
    y_max = int(min(original_data.shape[3], y+radius))
    x_min = int(max(0, x-radius))
    x_max = int(min(original_data.shape[4], x+radius))
    
    intensities = {}
    
    for marker_name, marker_info in MARKER_INDICES.items():
        marker_idx = marker_info
        threshold = MARKER_THRESHOLDS[marker_name]
        
        # Extract region data
        region_data = original_data[0, marker_idx, z_min:z_max, y_min:y_max, x_min:x_max]
        
        # Normalize data
        min_val = np.min(region_data)
        max_val = np.max(region_data)
        if max_val > min_val:
            normalized_data = (region_data - min_val) / (max_val - min_val)
        else:
            normalized_data = region_data
        
        # Filter by threshold (non-zero and above threshold)
        above_threshold = normalized_data[(normalized_data > 0) & (normalized_data > threshold)]
        
        if len(above_threshold) > 0:
            avg_intensity = np.mean(above_threshold)
            logger.debug(f"{marker_name} in region: {len(above_threshold)} voxels above threshold, avg: {avg_intensity:.4f}")
        else:
            avg_intensity = 0.0
            logger.debug(f"{marker_name} in region: no voxels above threshold")
            
        intensities[marker_name] = avg_intensity
    
    return intensities

def find_large_areas(segmentation_data, min_area_size=5):
    """Find large connected areas in segmentation"""
    labeled_areas, num_areas = label(segmentation_data)
    
    large_areas = []
    for i in range(1, num_areas + 1):
        # Use np.where instead of boolean mask to save memory
        coords = np.where(labeled_areas == i)
        area_size = len(coords[0])  # Number of coordinates = area size
        
        if area_size >= min_area_size:
            large_areas.append({
                'id': i,
                'size': area_size,
                'coords': coords
            })
    
    logger.info(f"Found {len(large_areas)} large areas (≥{min_area_size} voxels) from {num_areas} total areas")
    return large_areas

def calculate_weighted_center(coords, intensities):
    """Calculate weighted center based on intensity values for 3D data"""
    if len(coords[0]) == 0:
        return None
    
    # For 3D data, coords has 3 dimensions: (z, y, x)
    z_coords = coords[0]  # z coordinates
    y_coords = coords[1]  # y coordinates  
    x_coords = coords[2]  # x coordinates
    
    # Calculate weighted center
    total_weight = np.sum(intensities)
    if total_weight == 0:
        # Geometric center if no intensity
        center_z = np.mean(z_coords)
        center_y = np.mean(y_coords)
        center_x = np.mean(x_coords)
    else:
        # Weighted center - ensure intensities is 1D
        if intensities.ndim > 1:
            intensities = intensities.flatten()
        
        # Make sure all arrays have the same length
        min_length = min(len(z_coords), len(y_coords), len(x_coords), len(intensities))
        z_coords = z_coords[:min_length]
        y_coords = y_coords[:min_length]
        x_coords = x_coords[:min_length]
        intensities = intensities[:min_length]
        
        # Weighted center
        center_z = np.sum(z_coords * intensities) / total_weight
        center_y = np.sum(y_coords * intensities) / total_weight
        center_x = np.sum(x_coords * intensities) / total_weight
    
    # Return as integers
    return int(center_z), int(center_y), int(center_x)

def process_marker_areas(marker_name, original_data):
    """Process large areas for a specific marker"""
    logger.info(f"\nProcessing {marker_name}...")
    
    # Load segmentation
    backend_dir = Path(__file__).parent.resolve()
    input_dir = backend_dir / "input"
    seg_path = input_dir / f"segmentation_{marker_name}.zarr"
    
    if not seg_path.exists():
        logger.warning(f"Segmentation file not found: {seg_path}")
        return []
    
    # Load segmentation data
    seg_store = zarr.DirectoryStore(str(seg_path))
    seg_root = zarr.group(store=seg_store)
    segmentation = seg_root['segmentation'][:]
    
    # Find large areas
    large_areas = find_large_areas(segmentation, min_area_size=5)
    
    area_features = []
    for area in large_areas:
        # Calculate radius for this area
        radius = calculate_region_radius(area['size'])
        
        # Calculate weighted center using original marker data
        marker_idx = MARKER_INDICES[marker_name]
        coords = area['coords']
        intensities = original_data[0, marker_idx, coords[0], coords[1], coords[2]]
        weighted_center = calculate_weighted_center(coords, intensities)
        
        if weighted_center is None:
            continue
        
        # Get intensities of all markers in this region
        all_intensities = get_marker_intensities_in_region_with_threshold(weighted_center, radius, original_data)
        
        # Create feature row
        feature_row = {
            'area_id': area['id'],
            'marker': marker_name,
            'area_size': area['size'],
            'radius': radius,
            'z': int(round(weighted_center[0])),
            'y': int(round(weighted_center[1])),
            'x': int(round(weighted_center[2]))
        }
        
        # Add intensities for all markers
        for marker_name_intensity in MARKERS:
            feature_row[f'{marker_name_intensity}_intensity'] = round(all_intensities[marker_name_intensity], 4)
        
        area_features.append(feature_row)
    
    logger.info(f"Processed {len(area_features)} large areas for {marker_name}")
    return area_features

def merge_all_marker_features():
    """Merge features from all markers with weighted centers and non-zero averages"""
    logger.info("Starting feature merging process...")
    
    # Load original data
    original_data = load_original_data()
    if original_data is None:
        return
    
    all_area_features = []
    
    # Process each marker
    for marker in MARKERS:
        marker_features = process_marker_areas(marker, original_data)
        all_area_features.extend(marker_features)
    
    if not all_area_features:
        logger.error("No area features found!")
        return
    
    # Convert to DataFrame
    df = pd.DataFrame(all_area_features)
    
    # Save raw features
    output_path = "output/All_area_features.csv"
    df.to_csv(output_path, index=False)
    logger.info(f"Saved {len(df)} area features to {output_path}")
    
    # Create summary by marker
    logger.info("\nArea counts by marker:")
    for marker in MARKERS:
        marker_areas = df[df['marker'] == marker]
        logger.info(f"{marker}: {len(marker_areas)} areas")
        
        if len(marker_areas) > 0:
            avg_size = marker_areas['area_size'].mean()
            avg_intensity = marker_areas[f'{marker}_intensity'].mean()
            logger.info(f"  - Average area size: {avg_size:.1f} voxels")
            logger.info(f"  - Average intensity: {avg_intensity:.2f}")
    
    # Create normalized version with proper column order
    normalized_df = df.copy()
    
    # Normalize intensity values
    for marker in MARKERS:
        intensity_col = f'{marker}_intensity'
        if intensity_col in normalized_df.columns:
            max_val = normalized_df[intensity_col].max()
            if max_val > 0:
                normalized_df[intensity_col] = normalized_df[intensity_col] / max_val
                logger.info(f"Normalized {marker}: max value was {max_val:.2f}")
    
    # Reorganize columns to match the expected format
    # Expected format: cell_id, z, y, x, CD31, CD20, CD11b, CD4, CD11c, Catalase, marker
    final_df = pd.DataFrame()
    
    # Add cell_id (using area_id and marker)
    final_df['cell_id'] = normalized_df['marker'] + '_' + normalized_df['area_id'].astype(str)
    
    # Add z, y, x coordinates (rounded to nearest integer)
    final_df['z'] = normalized_df['z'].round().astype(int)
    final_df['y'] = normalized_df['y'].round().astype(int)
    final_df['x'] = normalized_df['x'].round().astype(int)
    
    # Add marker intensities in the correct order (rounded to 4 decimal places)
    final_df['CD31'] = normalized_df['CD31_intensity'].round(4)
    final_df['CD20'] = normalized_df['CD20_intensity'].round(4)
    final_df['CD11b'] = normalized_df['CD11b_intensity'].round(4)
    final_df['CD4'] = normalized_df['CD4_intensity'].round(4)
    final_df['CD11c'] = normalized_df['CD11c_intensity'].round(4)
    final_df['Catalase'] = normalized_df['Catalase_intensity'].round(4)
    
    # Add marker column
    final_df['marker'] = normalized_df['marker']
    
    # Sort by marker and then by cell_id
    final_df = final_df.sort_values(['marker', 'cell_id'])
    
    # Save normalized features with proper format
    norm_output_path = "output/Norm_all_cell_features.csv"
    final_df.to_csv(norm_output_path, index=False)
    logger.info(f"Saved {len(final_df)} normalized features to {norm_output_path}")
    
    # Log final statistics
    logger.info("\nFinal statistics:")
    logger.info(f"Total areas: {len(final_df)}")
    for marker in MARKERS:
        marker_count = len(final_df[final_df['marker'] == marker])
        logger.info(f"{marker}: {marker_count} areas")
    
    logger.info("\nFeature merging completed successfully!")

if __name__ == "__main__":
    merge_all_marker_features() 