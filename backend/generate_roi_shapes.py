import pandas as pd
import json
from pathlib import Path
import logging
from shapely.geometry import box

logging.basicConfig(level=logging.INFO)
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

def main():
    backend_dir = Path(__file__).parent.resolve()
    output_dir = backend_dir / "output"
    
    all_shapes = []
    
    # Process both CD31 and CD11b ROIs
    for marker in ["CD31", "CD11b"]:
        input_file = output_dir / f"roi_cells_{marker}_interactions.csv"
        if not input_file.exists():
            logger.warning(f"Input file not found: {input_file}")
            continue
            
        logger.info(f"Processing {input_file}")
        shapes = process_roi_csv(input_file)
        all_shapes.extend(shapes)
    
    # Create GeoJSON structure
    geojson = {
        "type": "FeatureCollection",
        "features": all_shapes
    }
    
    # Save to file
    output_file = output_dir / "roi_shapes.geojson"
    with open(output_file, 'w') as f:
        json.dump(geojson, f, indent=2)
    
    logger.info(f"Saved {len(all_shapes)} ROI shapes to {output_file}")

if __name__ == "__main__":
    main()
