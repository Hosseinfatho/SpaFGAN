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
        
        # Filter nearby ROIs, keeping the ones with higher scores
        filtered_rois = filter_nearby_rois(rois, radius=50)
        logger.info(f"Filtered {len(rois)} ROIs to {len(filtered_rois)} ROIs for {interaction_name}")
        
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
            segmentations[f"ROI_{idx:02d}"] = [polygon]
        
        roi_seg_path = output_dir / f"roi_segmentation_{safe_interaction_name}.json"
        with open(roi_seg_path, 'w') as f:
            json.dump(segmentations, f, indent=2)
        logger.info(f"Saved roi_segmentation_{safe_interaction_name}.json")
        
        # === Create obsFeatureMatrix_{interaction}.csv ===
        feature_data = []
        for idx, roi in enumerate(filtered_rois):
            feature_data.append({
                'roi_id': f"ROI_{idx:02d}",
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
            sets_data.append({
                'roi_id': f"ROI_{idx:02d}",
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
            roi_id = f"ROI_{idx:02d}"
            feature_matrix["data"][roi_id] = {
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
        # Apply filtering to combined data as well
        filtered_rois = filter_nearby_rois(rois, radius=25)
        all_rois.extend(filtered_rois)
    
    roi_df = pd.DataFrame(all_rois)
    
    # === Create combined obsSegmentations.json ===
    segmentations = {}
    y_max = 5508
    radius = 25 * 8  # Convert radius to pixel coordinates
    
    for idx, row in roi_df.iterrows():
        x = max(0, round(float(row['x']) * 8))
        y = max(0, round(y_max - float(row['y']) * 8))
        
        # Create circle polygon instead of rectangle
        polygon = create_circle_polygon(x, y, radius, num_points=32)
        polygon = [[max(0, round(coord[0])), max(0, round(coord[1]))] for coord in polygon]
        segmentations[f"ROI_{idx}"] = [polygon]

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

    for idx, row in roi_df.iterrows():
        roi_id = f"ROI_{idx}"
        feature_matrix["data"][roi_id] = {
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

if __name__ == "__main__":
    generate_vitessce_config()
