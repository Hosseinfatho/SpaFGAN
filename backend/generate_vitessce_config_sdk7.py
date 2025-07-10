import pandas as pd
import json
from pathlib import Path
import logging
import numpy as np
import glob

import yaml

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_roi_data(output_dir):
    roi_files = glob.glob(str(output_dir / "top_roi_scores_*.json"))
    all_rois = []

    for file_path in roi_files:
        with open(file_path, 'r') as f:
            data = json.load(f)
            interaction_name = data['interaction_name']
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
                all_rois.append(roi_data)

    return pd.DataFrame(all_rois)

def generate_vitessce_config():
    backend_dir = Path(__file__).parent.resolve()
    output_dir = backend_dir / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    roi_df = load_roi_data(output_dir)
    if roi_df.empty:
        logger.error("No ROI data found in JSON files")
        return

    # === Create obsSegmentations.json ===
    segmentations = {}
    y_max = 5508
    for idx, row in roi_df.iterrows():
        x = max(0, round(float(row['x']) * 8))  # Multiply by factor 8, round to int, and ensure >= 0
        y = max(0, round(y_max - float(row['y']) * 8))  # Multiply by factor 8, round to int, and ensure >= 0
        size = round(15 * 8)  # Multiply size by factor 8 and round to int
        polygon = [
            [x - size, y - size],
            [x + size, y - size],
            [x + size, y + size],
            [x - size, y + size],
            [x - size, y - size]
        ]

        # Ensure all coordinates are non-negative integers
        polygon = [[max(0, round(coord[0])), max(0, round(coord[1]))] for coord in polygon]

        segmentations[f"ROI_{idx}"] = [polygon]

    obs_seg_path = output_dir / "roi_rectangles_annotation.json"
    with open(obs_seg_path, 'w') as f:
        json.dump(segmentations, f, indent=2)

    logger.info(f"Saved obsSegmentations.json to {obs_seg_path}")

    # === Create obsFeatureMatrix.json ===
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

    logger.info(f"Saved obsFeatureMatrix.json to {obs_feat_path}")

    # === Print summary ===
    print("\n" + "="*50)
    print("VITESSCE CONFIGURATION SUMMARY")
    print("="*50)
    for interaction in roi_df['interaction'].unique():
        interaction_data = roi_df[roi_df['interaction'] == interaction]
        print(f"\n{interaction}:")
        print("-"*30)
        print(f"Number of ROIs: {len(interaction_data)}")
        print(f"Score range: {interaction_data['score'].max():.4f} - {interaction_data['score'].min():.4f}")
    print("\n" + "="*50)

if __name__ == "__main__":
    generate_vitessce_config()
