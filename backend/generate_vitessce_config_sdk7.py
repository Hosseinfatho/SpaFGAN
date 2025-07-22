import pandas as pd
import json
from pathlib import Path
import logging
import numpy as np
import glob
import re

import yaml

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
    """Create mapping from interaction to coarse category - updated"""
    mapping = {
        'B-cell infiltration': 'B-cell',
        'T-cell maturation': 'T-cell',
        'Inflammatory zone': 'Inflammatory',
        'Oxidative stress regulation': 'Oxidative'
    }
    return mapping

def categorize_score(score):
    """Categorize score into High, Medium, Low"""
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

def filter_corner_rois_simple(rois, min_x=50, min_y=50):
    """Filter ROIs that are in corners (x < 50 or y < 50)"""
    if not rois:
        return rois
    
    filtered_rois = []
    for roi in rois:
        x = roi.get('x', 0)
        y = roi.get('y', 0)
        
        # Check if ROI is not in corners (x >= 50 and y >= 50)
        if x >= min_x and y >= min_y:
            filtered_rois.append(roi)
    
    return filtered_rois



def calculate_distance(roi1, roi2):
    """Calculate Euclidean distance between two ROIs"""
    dx = roi1['x'] - roi2['x']
    dy = roi1['y'] - roi2['y']
    dz = roi1['z'] - roi2['z']
    return (dx**2 + dy**2 + dz**2)**0.5

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

def generate_vitessce_config():
    backend_dir = Path(__file__).parent.resolve()
    output_dir = backend_dir / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data grouped by interaction
    interaction_data = load_roi_data_by_interaction(output_dir)
    interaction_coarse = create_interaction_coarse_mapping()
    
    if not interaction_data:
        logger.error("No ROI data found in JSON files")
        return

    # Process each interaction separately
    for interaction_name, rois in interaction_data.items():
        logger.info(f"Processing interaction: {interaction_name}")
        
        # First filter ROIs that are in corners (x < 50 or y < 50)
        corner_filtered_rois = filter_corner_rois_simple(rois, min_x=50, min_y=50)
        logger.info(f"Corner filtered: {len(rois)} ROIs to {len(corner_filtered_rois)} ROIs for {interaction_name}")
        
        # Then filter nearby ROIs, keeping the ones with higher scores
        filtered_rois = filter_nearby_rois(corner_filtered_rois, radius=50)
        logger.info(f"Nearby filtered: {len(corner_filtered_rois)} ROIs to {len(filtered_rois)} ROIs for {interaction_name}")
        
        # Create safe filename
        safe_interaction_name = re.sub(r'[^a-zA-Z0-9_-]', '_', interaction_name)
        
        # === Create roi_segmentation_{interaction}.json ===
        segmentations = {}
        y_max = 5508
        radius = 25 * 8  # Convert radius to pixel coordinates (multiply by 8)
        
        for idx, roi in enumerate(filtered_rois):
            x = max(0, round(float(roi['x']) * 8))
            y = max(0, round(y_max - float(roi['y']) * 8))
            
            # Create circle polygon instead of rectangle
            polygon = create_circle_polygon(x, y, radius, num_points=32)
            polygon = [[max(0, round(coord[0])), max(0, round(coord[1]))] for coord in polygon]
            
            # Create ROI name with format: interaction_name_ID_score (ID starts from 1)
            roi_id = idx + 1  # Start from 1, not 0
            score = roi.get('score', 0)
            roi_name = f"{interaction_name}_{roi_id}_Score:{score:.3f}"
            segmentations[roi_name] = [polygon]
        
        roi_seg_path = output_dir / f"roi_segmentation_{safe_interaction_name}.json"
        with open(roi_seg_path, 'w') as f:
            json.dump(segmentations, f, indent=2)
        logger.info(f"Saved roi_segmentation_{safe_interaction_name}.json")
        
        # === Create obsFeatureMatrix_{interaction}.csv ===
        feature_data = []
        for idx, roi in enumerate(filtered_rois):
            roi_id = idx + 1  # Start from 1, not 0
            score = roi.get('score', 0)
            roi_name = f"{interaction_name}_{roi_id}_Score:{score:.3f}"
            feature_data.append({
                'roi_id': roi_name,
                'score': roi['score'],
                'intensity_score': roi['intensity_score'],
                'attention_score': roi['attention_score'],
                'num_nodes': roi['num_nodes'],
                'num_edges': roi['num_edges'],
                'x': roi['x'],
                'y': roi['y'],
                'z': roi['z']
            })
        
        feature_df = pd.DataFrame(feature_data)
        feature_path = output_dir / f"obsFeatureMatrix_{safe_interaction_name}.csv"
        feature_df.to_csv(feature_path, index=False)
        logger.info(f"Saved obsFeatureMatrix_{safe_interaction_name}.csv")
        
        # === Create obsSets_{interaction}.csv ===
        sets_data = []
        for idx, roi in enumerate(filtered_rois):
            roi_id = idx + 1  # Start from 1, not 0
            score = roi.get('score', 0)
            roi_name = f"{interaction_name}_{roi_id}_Score:{score:.3f}"
            sets_data.append({
                'roi_id': roi_name,
                'interaction': interaction_name,
                'interaction_coarse': interaction_coarse.get(interaction_name, 'Unknown'),
                'score_category': categorize_score(roi['score']),
                'pred_interaction': interaction_name,
                'pred_score': roi['score']
            })
        
        sets_df = pd.DataFrame(sets_data)
        sets_path = output_dir / f"obsSets_{safe_interaction_name}.csv"
        sets_df.to_csv(sets_path, index=False)
        logger.info(f"Saved obsSets_{safe_interaction_name}.csv")
        
        # === Create obsFeatureMatrix_{interaction}.json ===
        feature_matrix = {
            "version": "0.1.0",
            "type": "obsFeatureMatrix",
            "data": {}
        }
        
        for idx, roi in enumerate(filtered_rois):
            roi_id = idx + 1  # Start from 1, not 0
            score = roi.get('score', 0)
            roi_name = f"{interaction_name}_{roi_id}_Score:{score:.3f}"
            feature_matrix["data"][roi_name] = {
                "score": float(roi['score']),
                "interaction": roi['interaction'],
                "intensity_score": float(roi['intensity_score']),
                "attention_score": float(roi['attention_score']),
                "num_nodes": int(roi['num_nodes']),
                "num_edges": int(roi['num_edges'])
            }
        
        feature_json_path = output_dir / f"obsFeatureMatrix_{safe_interaction_name}.json"
        with open(feature_json_path, 'w') as f:
            json.dump(feature_matrix, f, indent=2)
        logger.info(f"Saved obsFeatureMatrix_{safe_interaction_name}.json")

    # === Create combined files (original functionality) ===
    all_rois = []
    for interaction_name, rois in interaction_data.items():
        # Apply corner filtering first (x < 50 or y < 50)
        corner_filtered_rois = filter_corner_rois_simple(rois, min_x=50, min_y=50)
        # Then apply nearby filtering
        filtered_rois = filter_nearby_rois(corner_filtered_rois, radius=25)
        all_rois.extend(filtered_rois)
    
    roi_df = pd.DataFrame(all_rois)
    
    # === Create combined obsSegmentations.json ===
    segmentations = {}
    y_max = 5508
    radius = 25 * 8  # Convert radius to pixel coordinates
    
    # Create counters for each interaction to start from 1
    interaction_counters = {}
    
    for idx, row in roi_df.iterrows():
        x = max(0, round(float(row['x']) * 8))
        y = max(0, round(y_max - float(row['y']) * 8))
        
        # Create circle polygon instead of rectangle
        polygon = create_circle_polygon(x, y, radius, num_points=32)
        polygon = [[max(0, round(coord[0])), max(0, round(coord[1]))] for coord in polygon]
        
        # Create ROI name with format: interaction_name_ID_score (ID starts from 1)
        interaction_name = row['interaction']
        if interaction_name not in interaction_counters:
            interaction_counters[interaction_name] = 0
        interaction_counters[interaction_name] += 1
        roi_id = interaction_counters[interaction_name]
        score = row.get('score', 0)
        roi_name = f"{interaction_name}_{roi_id}_Score:{score:.3f}"
        segmentations[roi_name] = [polygon]

    obs_seg_path = output_dir / "roi_rectangles_annotation.json"
    with open(obs_seg_path, 'w') as f:
        json.dump(segmentations, f, indent=2)
    logger.info(f"Saved combined obsSegmentations.json to {obs_seg_path}")

    # === Create combined obsFeatureMatrix.json ===
    feature_matrix = {
        "version": "0.1.0",
        "type": "obsFeatureMatrix",
        "data": {}
    }

    # Create counters for each interaction to start from 1
    interaction_counters = {}
    
    for idx, row in roi_df.iterrows():
        interaction_name = row['interaction']
        if interaction_name not in interaction_counters:
            interaction_counters[interaction_name] = 0
        interaction_counters[interaction_name] += 1
        roi_id = interaction_counters[interaction_name]
        score = row.get('score', 0)
        roi_name = f"{interaction_name}_{roi_id}_Score:{score:.3f}"
        feature_matrix["data"][roi_name] = {
            "score": float(row['score']),
            "interaction": row['interaction'],
            "intensity_score": float(row['intensity_score']),
            "attention_score": float(row['attention_score']),
            "num_nodes": int(row['num_nodes']),
            "num_edges": int(row['num_edges'])
        }

    obs_feat_path = output_dir / "obsFeatureMatrix.json"
    with open(obs_feat_path, 'w') as f:
        json.dump(feature_matrix, f, indent=2)
    logger.info(f"Saved combined obsFeatureMatrix.json to {obs_feat_path}")

    # === Print summary ===
    print("\n" + "="*60)
    print("INTERACTION-SPECIFIC FILES GENERATION SUMMARY")
    print("="*60)
    for interaction_name, rois in interaction_data.items():
        safe_name = re.sub(r'[^a-zA-Z0-9_-]', '_', interaction_name)
        filtered_rois = filter_nearby_rois(rois, radius=25)
        print(f"\n{interaction_name}:")
        print("-" * 40)
        print(f"Original ROIs: {len(rois)}")
        print(f"Filtered ROIs: {len(filtered_rois)}")
        print(f"Score range: {max(r['score'] for r in filtered_rois):.4f} - {min(r['score'] for r in filtered_rois):.4f}")
        print(f"Files created:")
        print(f"  - roi_segmentation_{safe_name}.json")
        print(f"  - obsFeatureMatrix_{safe_name}.csv")
        print(f"  - obsFeatureMatrix_{safe_name}.json")
        print(f"  - obsSets_{safe_name}.csv")
    print("\n" + "="*60)

def update_roi_json_format():
    """Update ROI JSON files to new format with only top 7 ROIs and tooltip-friendly format"""
    
    output_dir = Path(__file__).parent / 'output'
    
    # List of interaction names
    interaction_names = [
        'B-cell infiltration',
        'T-cell maturation', 
        'Inflammatory zone',
        'Oxidative stress regulation'
    ]
    
    for interaction_name in interaction_names:
        input_file = output_dir / f'top_roi_scores_{interaction_name}.json'
        
        if not input_file.exists():
            logger.info(f"File not found: {input_file}")
            continue
            
        logger.info(f"Processing: {input_file}")
        
        # Read the original file
        with open(input_file, 'r') as f:
            data = json.load(f)
        
        # Get top 7 ROIs
        top_rois = data.get('top_rois', [])[:7]
        
        # Create new format
        new_data = {
            "interaction_name": interaction_name,
            "top_rois": []
        }
        
        for roi in top_rois:
            roi_id = roi.get('roi_id', 0)
            score = roi.get('scores', {}).get('combined_score', 0)
            
            # Create tooltip-friendly name
            tooltip_name = f"{interaction_name}_{roi_id}_Score:{score:.3f}"
            
            new_roi = {
                "roi_id": roi_id,
                "interaction": interaction_name,
                "tooltip_name": tooltip_name,
                "position": roi.get('position', {}),
                "scores": roi.get('scores', {}),
                "num_nodes": roi.get('num_nodes', 0),
                "num_edges": roi.get('num_edges', 0)
            }
            
            new_data["top_rois"].append(new_roi)
        
        # Write the updated file
        with open(input_file, 'w') as f:
            json.dump(new_data, f, indent=2)
        
        logger.info(f"Updated {input_file} with {len(new_data['top_rois'])} ROIs")

if __name__ == "__main__":
    # Update ROI JSON format first
    update_roi_json_format()
    
    # Then generate Vitessce config
    generate_vitessce_config()
