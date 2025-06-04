import pandas as pd
import json
from pathlib import Path
import logging
from shapely.geometry import box
import numpy as np
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_cube_shape(center_x, center_y, z_min, z_max, size=50, factor=8):
    """
    Create a cube shape centered at (center_x, center_y) with ±size range in x and y,
    and full z range. x and y are multiplied by factor for real size.
    
    Args:
        center_x (float): Center x coordinate
        center_y (float): Center y coordinate
        z_min (float): Minimum z value
        z_max (float): Maximum z value
        size (float): Half-width of the cube in x and y directions
        factor (float): Scaling factor for real size
        
    Returns:
        dict: GeoJSON feature with cube shape
    """
    # Apply scaling
    center_x *= factor
    center_y *= factor
    min_x = center_x - size * factor
    max_x = center_x + size * factor
    min_y = center_y - size * factor
    max_y = center_y + size * factor
    
    # Create box shape
    cube = box(min_x, min_y, max_x, max_y)
    
    return {
        "type": "Feature",
        "geometry": {
            "type": "Polygon",
            "coordinates": [[
                [min_x, min_y],
                [max_x, min_y],
                [max_x, max_y],
                [min_x, max_y],
                [min_x, min_y]
            ]]
        },
        "properties": {
            "z_range": [z_min, z_max],
            "center": [center_x, center_y],
            "size": size * factor
        }
    }

def process_roi_csv(csv_path):
    try:
        df = pd.read_csv(csv_path)
        if df.empty or df.columns.size == 0:
            logger.warning(f" Skipping empty or invalid file: {csv_path.name}")
            return []
    except pd.errors.EmptyDataError:
        logger.warning(f" Skipping corrupt file: {csv_path.name}")
        return []

    shapes = []
    
    # Get z range for the entire dataset
    z_min = df['z'].min()
    z_max = df['z'].max()

    for roi_id, group in df.groupby("roi_id"):
        # Get center coordinates
        center_x = group['x'].mean()
        center_y = group['y'].mean()
        
        # Get interaction scores
        interactions = []
        for col in group.columns:
            if col in ['T-cell_entry_site', 'B-cell_infiltration', 'Oxidative_stress_niche', 
                      'Inflammatory_zone', 'Dendritic_signal']:
                if group[col].mean() > 0.5:  # Threshold for interaction
                    interactions.append(col)
        
        # Create cube shape
        shape = create_cube_shape(center_x, center_y, z_min, z_max)
        
        # Add ROI properties
        shape["properties"].update({
            "name": roi_id,
            "type": "ROI",
            "score": float(round(group["spafgan_score"].mean(), 3)),
            "interactions": interactions,
            "marker_values": {
                marker: float(group[marker].mean())
                for marker in ['CD31', 'CD4', 'CD20', 'CD11b', 'CD11c', 'Catalase']
                if marker in group.columns
            }
        })
        
        shapes.append(shape)

    return shapes

def calculate_roi_shape(points):
    """
    Calculate shape properties of ROI using convex hull
    """
    try:
        hull = ConvexHull(points)
        return {
            'area': hull.volume,  # Area in 2D
            'perimeter': hull.area,  # Perimeter in 2D
            'vertices': len(hull.vertices),
            'centroid_x': np.mean(points[:, 0]),
            'centroid_y': np.mean(points[:, 1])
        }
    except:
        return {
            'area': 0,
            'perimeter': 0,
            'vertices': 0,
            'centroid_x': np.mean(points[:, 0]),
            'centroid_y': np.mean(points[:, 1])
        }

def print_roi_summary(shapes_df):
    """
    Print summary of ROI statistics for each marker
    """
    print("\n" + "="*50)
    print("ROI SUMMARY STATISTICS")
    print("="*50)
    
    for marker in ["CD31", "CD11b", "CD11c"]:
        marker_data = shapes_df[shapes_df['marker'] == marker]
        if len(marker_data) > 0:
            print(f"\n{marker}:")
            print("-"*30)
            print(f"Total ROIs: {len(marker_data)}")
            
            # Group by interaction
            for interaction in marker_data['interaction'].unique():
                interaction_data = marker_data[marker_data['interaction'] == interaction]
                min_score = interaction_data['avg_score'].min()
                max_score = interaction_data['avg_score'].max()
                num_cells = interaction_data['num_cells'].sum()
                
                print(f"\n{interaction}:")
                print(f"  Number of cells: {num_cells}")
                print(f"  Score range: {min_score:.3f} - {max_score:.3f}")
                print(f"  Average area: {interaction_data['area'].mean():.2f}")
                print(f"  Average perimeter: {interaction_data['perimeter'].mean():.2f}")
    
    print("\n" + "="*50)

def calculate_global_stats(rois_df):
    """Calculate global statistics for each interaction"""
    stats = {}
    for interaction in rois_df['interaction'].unique():
        interaction_data = rois_df[rois_df['interaction'] == interaction]
        stats[interaction] = {
            'min': interaction_data['score'].min(),
            'max': interaction_data['score'].max(),
            'mean': interaction_data['score'].mean(),
            'std': interaction_data['score'].std()
        }
    return stats

def normalize_score(score, stats, interaction):
    """Normalize score based on global statistics"""
    if interaction not in stats:
        return score
        
    interaction_stats = stats[interaction]
    min_val = interaction_stats['min']
    max_val = interaction_stats['max']
    
    if max_val == min_val:
        return 0.5
    
    # Add small epsilon to prevent multiple 1.0 scores
    epsilon = 1e-6
    normalized = (score - min_val) / (max_val - min_val + epsilon)
    
    # Round to 4 decimal places for more granularity
    return round(max(0.0, min(1.0, normalized)), 4)

def is_far_enough(x, y, existing_rois, min_distance=50):
    """
    Check if a point is far enough from existing ROIs
    
    Args:
        x (float): x coordinate
        y (float): y coordinate
        existing_rois (list): List of existing ROIs with x,y coordinates
        min_distance (float): Minimum required distance
        
    Returns:
        bool: True if point is far enough from all existing ROIs
    """
    for roi in existing_rois:
        distance = np.sqrt((x - roi['x'])**2 + (y - roi['y'])**2)
        if distance < min_distance:
            return False
    return True

def generate_roi_shapes(output_dir):
    """
    Sort ROIs based on interaction scores and save top 10% for Vitnesse configuration,
    ensuring minimum distance between ROIs
    """
    all_rois = []
    existing_rois = []  # Keep track of added ROIs for distance checking
    
    # First pass: collect all ROIs
    for marker in ["CD31", "CD11b", "CD11c"]:
        roi_path = output_dir / f"extracted_rois_{marker}.csv"
        if not roi_path.exists():
            logger.warning(f"ROI file not found for {marker}, skipping...")
            continue
            
        # Load ROI data
        roi_df = pd.read_csv(roi_path)
        
        # Process each interaction
        for interaction in roi_df.columns:
            if interaction in ['cell_id', 'x', 'y', 'z', 'interactions']:
                continue
                
            # Get cells with this interaction
            interaction_cells = roi_df[roi_df[interaction] > 0.3].copy()
            if len(interaction_cells) > 0:
                # Sort by interaction score and get top 10%
                n_top = max(1, int(len(interaction_cells) * 0.1))  # At least 1 cell
                top_cells = interaction_cells.nlargest(n_top, interaction)
                
                # Add to ROIs list, checking distance
                for _, row in top_cells.iterrows():
                    if is_far_enough(row['x'], row['y'], existing_rois):
                        roi_data = {
                            'marker': marker,
                            'interaction': interaction,
                            'x': row['x'],
                            'y': row['y'],
                            'score': row[interaction]
                        }
                        all_rois.append(roi_data)
                        existing_rois.append({'x': row['x'], 'y': row['y']})
    
    # Create final dataframe and sort
    final_df = pd.DataFrame(all_rois)
    final_df = final_df.sort_values(['marker', 'interaction', 'score'], 
                                  ascending=[True, True, False])
    
    output_path = output_dir / "roi_data_for_vitnesse.csv"
    final_df.to_csv(output_path, index=False)
    
    # Print summary
    print("\n" + "="*50)
    print("TOP 10% ROI DATA SUMMARY FOR VITNESSE")
    print("="*50)
    
    for marker in ["CD31", "CD11b", "CD11c"]:
        marker_data = final_df[final_df['marker'] == marker]
        if len(marker_data) > 0:
            print(f"\n{marker}:")
            print("-"*30)
            for interaction in marker_data['interaction'].unique():
                interaction_data = marker_data[marker_data['interaction'] == interaction]
                print(f"{interaction}: {len(interaction_data)} cells")
                print(f"Score range: {interaction_data['score'].max():.4f} - {interaction_data['score'].min():.4f}")
    
    print("\n" + "="*50)
    logger.info(f"Saved top 10% ROI data to {output_path}")
    
    # Print distance statistics
    print("\nDistance Statistics:")
    print("-"*30)
    distances = []
    for i in range(len(existing_rois)):
        for j in range(i+1, len(existing_rois)):
            roi1 = existing_rois[i]
            roi2 = existing_rois[j]
            distance = np.sqrt((roi1['x'] - roi2['x'])**2 + (roi1['y'] - roi2['y'])**2)
            distances.append(distance)
    
    if distances:
        print(f"Minimum distance: {min(distances):.2f}")
        print(f"Maximum distance: {max(distances):.2f}")
        print(f"Average distance: {np.mean(distances):.2f}")
        print(f"Total ROIs: {len(existing_rois)}")

def main():
    backend_dir = Path(__file__).parent.resolve()
    output_dir = backend_dir / "output"
    generate_roi_shapes(output_dir)

if __name__ == "__main__":
    main()
