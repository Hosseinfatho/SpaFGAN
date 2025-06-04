import pandas as pd
import json
from pathlib import Path
import logging
import numpy as np

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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
        "markers": {
            "CD31": {"color": "#FF0000", "shape": "circle"},
            "CD11b": {"color": "#0000FF", "shape": "square"},
            "CD11c": {"color": "#00FF00", "shape": "triangle"}
        },
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
            "id": f"{row['marker']}_{row['interaction']}_{len(config['rois'])}",
            "marker": row['marker'],
            "interaction": row['interaction'],
            "position": {
                "x": float(row['x']),
                "y": float(row['y'])
            },
            "score": float(row['score']),
            "properties": {
                "marker": row['marker'],
                "interaction": row['interaction'],
                "score": float(row['score'])
            }
        }
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
    
    for marker in ["CD31", "CD11b", "CD11c"]:
        marker_data = roi_df[roi_df['marker'] == marker]
        if len(marker_data) > 0:
            print(f"\n{marker}:")
            print("-"*30)
            for interaction in marker_data['interaction'].unique():
                interaction_data = marker_data[marker_data['interaction'] == interaction]
                print(f"{interaction}: {len(interaction_data)} ROIs")
                print(f"Score range: {interaction_data['score'].max():.4f} - {interaction_data['score'].min():.4f}")
    
    print("\n" + "="*50)

def generate_vitnesse_config():
    # Read ROI data
    roi_path = Path(__file__).parent / "output" / "roi_data_for_vitnesse.csv"
    df = pd.read_csv(roi_path)
    
    # Get image dimensions for Y flip
    y_max = 5508  # Full height of the image
    
    # Create GeoJSON structure
    geojson = {
        "type": "FeatureCollection",
        "features": []
    }
    
    # Convert each row to a GeoJSON feature
    for _, row in df.iterrows():
        # Create a polygon around the point (50x50 pixels)
        x = float(row['x'])
        y = y_max - float(row['y'])  # Flip Y coordinate
        size = 25  # half of the desired size
        
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
                "interactions": [row['interaction']],  # Make it a list as expected by ROISelector
                "marker_values": {
                    row['marker']: float(row['score'])
                }
            }
        }
        geojson["features"].append(feature)
    
    # Save as GeoJSON
    output_path = Path(__file__).parent / "output" / "roi_shapes.geojson"
    with open(output_path, 'w') as f:
        json.dump(geojson, f, indent=2)
    
    print(f"Generated {len(geojson['features'])} ROIs in GeoJSON format")
    print(f"Saved to: {output_path}")
    
    # Also save the Vitnesse config
    create_vitnesse_config(df, Path(__file__).parent / "output")

def main():
    backend_dir = Path(__file__).parent.resolve()
    output_dir = backend_dir / "output"
    
    # Load ROI data
    roi_path = output_dir / "roi_data_for_vitnesse.csv"
    if not roi_path.exists():
        logger.error(f"ROI data file not found: {roi_path}")
        return
    
    roi_df = pd.read_csv(roi_path)
    create_vitnesse_config(roi_df, output_dir)

if __name__ == "__main__":
    generate_vitnesse_config() 