import pandas as pd
import json
from pathlib import Path
import logging
import numpy as np
import glob
import re
import zarr
import spatialdata as sd

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_roi_data_by_interaction(output_dir):
    """Load ROI data grouped by interaction"""
    roi_files = glob.glob(str(output_dir / "top_roi_scores_*.json"))
    interaction_data = {}
    
    for file_path in roi_files:
        with open(file_path, 'r') as f:
            data = json.load(f)
            interaction_name = data['interaction_name']
            interaction_data[interaction_name] = []
            
            for roi in data['top_rois']:
                roi_data = {
                    'x': roi['position']['x'],
                    'y': roi['position']['y'],
                    'z': roi['position'].get('z', 0),
                    'score': roi['scores']['combined_score'],
                    'interaction': interaction_name,
                    'roi_id': roi['roi_id'],
                    'intensity_score': roi['scores']['intensity_score'],
                    'attention_score': roi['scores']['attention_score'],
                    'num_nodes': roi['num_nodes'],
                    'num_edges': roi['num_edges']
                }
                interaction_data[interaction_name].append(roi_data)
    
    return interaction_data

def create_interaction_coarse_mapping():
    """Create mapping from interaction to coarse category"""
    mapping = {
        'B-cell infiltration': 'B-cell',
        'T-cell entry site': 'T-cell',
        'Inflammatory zone': 'Inflammatory',
        'Oxidative stress niche': 'Oxidative',
        'Dendritic signal': 'Dendritic'
    }
    return mapping

def categorize_score(score):
    """Categorize score into High, Medium, Low - NOT USED, keeping for compatibility"""
    if score >= 0.5:
        return 'High'
    elif score >= 0.3:
        return 'Medium'
    else:
        return 'Low'

def filter_nearby_rois(rois, radius=25):
    """Filter ROIs that are too close to each other, keeping the one with higher score"""
    if not rois:
        return rois
    
    # Sort ROIs by score in descending order (highest first)
    sorted_rois = sorted(rois, key=lambda x: x['score'], reverse=True)
    filtered_rois = []
    
    for current_roi in sorted_rois:
        # Check if current ROI is too close to any already accepted ROI
        too_close = False
        for accepted_roi in filtered_rois:
            distance = calculate_distance(current_roi, accepted_roi)
            if distance < radius:
                too_close = True
                break
        
        # If not too close to any accepted ROI, add it
        if not too_close:
            filtered_rois.append(current_roi)
    
    return filtered_rois

def calculate_distance(roi1, roi2):
    """Calculate Euclidean distance between two ROIs"""
    dx = roi1['x'] - roi2['x']
    dy = roi1['y'] - roi2['y']
    dz = roi1['z'] - roi2['z']
    return (dx**2 + dy**2 + dz**2)**0.5

def transform_coordinates(x, y, factor=8, y_max=5508):
    """Transform coordinates: multiply by factor and flip y coordinate"""
    x_transformed = round(float(x) * factor)
    y_transformed = round(y_max - float(y) * factor)
    return x_transformed, y_transformed

def round_score(score, decimals=3):
    """Round score to specified number of decimal places (precision 0.001)"""
    return round(float(score), decimals)

def create_circle_polygon(center_x, center_y, radius, num_points=32):
    """Create a circle polygon with specified number of points"""
    import math
    polygon = []
    for i in range(num_points + 1):  # +1 to close the circle
        angle = 2 * math.pi * i / num_points
        x = center_x + radius * math.cos(angle)
        y = center_y + radius * math.sin(angle)
        polygon.append([round(x), round(y)])
    return polygon

def generate_annodata_zarr():
    """Generate AnnData Zarr file from ROI data with exact structure matching existing file"""
    backend_dir = Path(__file__).parent.resolve()
    output_dir = backend_dir / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data grouped by interaction
    interaction_data = load_roi_data_by_interaction(output_dir)
    interaction_coarse = create_interaction_coarse_mapping()
    
    if not interaction_data:
        logger.error("No ROI data found in JSON files")
        return

    # Collect all ROIs with coordinate transformation and score rounding
    all_rois = []
    factor = 8
    y_max = 5508
    radius = 100  # Fixed radius as seen in existing file
    
    for interaction_name, rois in interaction_data.items():
        logger.info(f"Processing interaction: {interaction_name}")
        
        # Filter nearby ROIs, keeping the ones with higher scores
        filtered_rois = filter_nearby_rois(rois, radius=50)
        logger.info(f"Filtered {len(rois)} ROIs to {len(filtered_rois)} ROIs for {interaction_name}")
        
        for idx, roi in enumerate(filtered_rois):
            # Transform coordinates
            x_transformed, y_transformed = transform_coordinates(roi['x'], roi['y'], factor, y_max)
            
            # Round scores to 3 decimal places
            score_rounded = round_score(roi['score'])
            intensity_score_rounded = round_score(roi['intensity_score'])
            attention_score_rounded = round_score(roi['attention_score'])
            
            roi_data = {
                'roi_id': roi['roi_id'],  # Keep original ROI ID
                'x': x_transformed,
                'y': y_transformed,
                'z': round(float(roi['z'])),
                'combined_score': score_rounded,
                'intensity_score': intensity_score_rounded,
                'attention_score': attention_score_rounded,
                'interaction': interaction_name,
                'radius': radius
            }
            all_rois.append(roi_data)
    
    logger.info(f"Total ROIs after processing: {len(all_rois)}")
    
    # Create Zarr file with exact structure
    try:
        import zarr
        
        # Create root group
        zarr_path = output_dir / "test.zarr"
        root = zarr.open_group(str(zarr_path), mode='w')
        
        # Add root attributes
        root.attrs['spatialdata_attrs'] = {
            'spatialdata_software_version': '0.4.0',
            'version': '0.1'
        }
        
        # Create table structure
        table_group = root.create_group('table/table')
        
        # Get data arrays
        roi_ids = [roi['roi_id'] for roi in all_rois]
        x_coords = [roi['x'] for roi in all_rois]
        y_coords = [roi['y'] for roi in all_rois]
        combined_scores = [roi['combined_score'] for roi in all_rois]
        intensity_scores = [roi['intensity_score'] for roi in all_rois]
        attention_scores = [roi['attention_score'] for roi in all_rois]
        interactions = [roi['interaction'] for roi in all_rois]
        
        # Create table arrays
        table_group.create_dataset('roi_id', data=roi_ids, dtype='<U10')
        table_group.create_dataset('x', data=x_coords, dtype='<i8')
        table_group.create_dataset('y', data=y_coords, dtype='<i8')
        table_group.create_dataset('combined_score', data=combined_scores, dtype='<f8')
        table_group.create_dataset('intensity_score', data=intensity_scores, dtype='<f8')
        table_group.create_dataset('attention_score', data=attention_scores, dtype='<f8')
        table_group.create_dataset('interaction', data=interactions, dtype='<U25')
        
        # Create shapes structure
        shapes_group = root.create_group('shapes/roi_circles')
        
        # Add shapes attributes
        shapes_group.attrs['axes'] = ['x', 'y']
        shapes_group.attrs['coordinateTransformations'] = [{
            'input': {
                'axes': [
                    {'name': 'x', 'type': 'space', 'unit': 'unit'},
                    {'name': 'y', 'type': 'space', 'unit': 'unit'}
                ],
                'name': 'xy'
            },
            'output': {
                'axes': [
                    {'name': 'x', 'type': 'space', 'unit': 'unit'},
                    {'name': 'y', 'type': 'space', 'unit': 'unit'}
                ],
                'name': 'global'
            },
            'type': 'identity'
        }]
        shapes_group.attrs['encoding-type'] = 'ngff:shapes'
        shapes_group.attrs['spatialdata_attrs'] = {'version': '0.2'}
        
        # Create shapes arrays (same data as table)
        shapes_group.create_dataset('roi_id', data=roi_ids, dtype='<U10')
        shapes_group.create_dataset('x', data=x_coords, dtype='<i8')
        shapes_group.create_dataset('y', data=y_coords, dtype='<i8')
        shapes_group.create_dataset('combined_score', data=combined_scores, dtype='<f8')
        shapes_group.create_dataset('intensity_score', data=intensity_scores, dtype='<f8')
        shapes_group.create_dataset('attention_score', data=attention_scores, dtype='<f8')
        shapes_group.create_dataset('interaction', data=interactions, dtype='<U25')
        shapes_group.create_dataset('radius', data=[radius] * len(all_rois), dtype='<i4')
        
        logger.info(f"Successfully created Zarr file: {zarr_path}")
        
        # Print summary
        print("\n" + "="*60)
        print("ANNODATA ZARR GENERATION SUMMARY")
        print("="*60)
        print(f"Total ROIs processed: {len(all_rois)}")
        print(f"Coordinate transformation factor: {factor}")
        print(f"Y-axis maximum: {y_max}")
        print(f"Score rounding: 3 decimal places")
        print(f"Radius: {radius}")
        print(f"Files created:")
        print(f"  - roi_shapes.spatialdata.zarr")
        
        # Print interaction summary
        interaction_counts = {}
        for roi in all_rois:
            interaction_counts[roi['interaction']] = interaction_counts.get(roi['interaction'], 0) + 1
        
        print(f"\nROI counts by interaction:")
        for interaction, count in interaction_counts.items():
            print(f"  - {interaction}: {count}")
        
        # Print score statistics
        scores = [roi['combined_score'] for roi in all_rois]
        print(f"\nScore statistics:")
        print(f"  - Min score: {min(scores):.3f}")
        print(f"  - Max score: {max(scores):.3f}")
        print(f"  - Mean score: {sum(scores)/len(scores):.3f}")
        
        print("\n" + "="*60)
        
    except Exception as e:
        logger.error(f"Error creating Zarr file: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    generate_annodata_zarr() 