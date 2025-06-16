import pandas as pd
import json
from pathlib import Path
import logging
import numpy as np
import glob

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_roi_data(output_dir):
    """
    Load ROI data from JSON files
    
    Args:
        output_dir (Path): Directory containing ROI JSON files
    
    Returns:
        pd.DataFrame: Combined ROI data
    """
    roi_files = glob.glob(str(output_dir / "extraction_roi_*.json"))
    all_rois = []
    
    for file_path in roi_files:
        with open(file_path, 'r') as f:
            data = json.load(f)
            interaction_name = data['interaction_name']
            for roi in data['rois']:
                roi_data = {
                    'x': roi['x'],
                    'y': roi['y'],
                    'z': roi.get('z', 0),  # Optional z coordinate
                    'score': roi['score'],
                    'interaction': interaction_name
                }
                all_rois.append(roi_data)
    
    return pd.DataFrame(all_rois)

def create_vitnesse_config(roi_df, output_dir):
    """
    Create Vitnesse configuration from ROI data
    
    Args:
        roi_df (pd.DataFrame): DataFrame containing ROI data
        output_dir (Path): Directory to save configuration
    """
    # Create base configuration
    config = {
        "version": "1.0",
        "name": "SpaFGAN ROIs",
        "description": "ROIs generated from SpaFGAN analysis",
        "interactions": {
            "T-cell entry site": {"color": "#FF0000"},
            "Inflammatory zone": {"color": "#0000FF"},
            "Oxidative stress niche": {"color": "#00FF00"},
            "B-cell infiltration": {"color": "#800080"},
            "Dendritic signal": {"color": "#FFA500"}
        },
        "rois": []
    }
    
    # Process each ROI
    for _, row in roi_df.iterrows():
        roi_config = {
            "id": f"{row['interaction']}_{len(config['rois'])}",
            "interaction": row['interaction'],
            "position": {
                "x": float(row['x']),
                "y": float(row['y'])
            },
            "score": float(row['score']),
            "properties": {
                "interaction": row['interaction'],
                "score": float(row['score'])
            }
        }
        
        # Add z coordinate if available
        if 'z' in row and not pd.isna(row['z']):
            roi_config['position']['z'] = float(row['z'])
            roi_config['properties']['z'] = float(row['z'])
            
        config['rois'].append(roi_config)
    
    # Save configuration
    output_path = output_dir / "vitnesse_config.json"
    with open(output_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    logger.info(f"Saved Vitnesse configuration to {output_path}")
    
    # Print summary
    print("\n" + "="*50)
    print("VITNESSE CONFIGURATION SUMMARY")
    print("="*50)
    
    for interaction in roi_df['interaction'].unique():
        interaction_data = roi_df[roi_df['interaction'] == interaction]
        print(f"\n{interaction}:")
        print("-"*30)
        print(f"Number of ROIs: {len(interaction_data)}")
        print(f"Score range: {interaction_data['score'].max():.4f} - {interaction_data['score'].min():.4f}")
    
    print("\n" + "="*50)

def generate_vitnesse_config():
    # Setup paths
    backend_dir = Path(__file__).parent.resolve()
    output_dir = backend_dir / "output"
    
    # Load ROI data from JSON files
    roi_df = load_roi_data(output_dir)
    if roi_df.empty:
        logger.error("No ROI data found in JSON files")
        return
    
    # Create Vitnesse configuration
    create_vitnesse_config(roi_df, output_dir)
    
    # Create GeoJSON structure
    geojson = {
        "type": "FeatureCollection",
        "features": []
    }
    
    # Get image dimensions for Y flip
    y_max = 5508  # Full height of the image
    
    # Convert each row to a GeoJSON feature
    for _, row in roi_df.iterrows():
        # Create a polygon around the point (50x50 pixels)
        x = float(row['x']) * 8  # Multiply x by 8
        y = y_max - (float(row['y']) * 8)  # Multiply y by 8 then flip
        size = 30  # half of the desired size
        
        feature = {
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": [[
                    [x - size, y - size],
                    [x + size, y - size],
                    [x + size, y + size],
                    [x - size, y + size],
                    [x - size, y - size]
                ]]
            },
            "properties": {
                "name": f"ROI_{row.name}",
                "score": float(row['score']),
                "interactions": [row['interaction']],
                "marker_values": {
                    row['interaction']: float(row['score'])
                }
            }
        }
        geojson["features"].append(feature)
    
    # Save as GeoJSON
    output_path = output_dir / "roi_shapes.geojson"
    with open(output_path, 'w') as f:
        json.dump(geojson, f, indent=2)
    
    print(f"Generated {len(geojson['features'])} ROIs in GeoJSON format")
    print(f"Saved to: {output_path}")

if __name__ == "__main__":
    generate_vitnesse_config() 