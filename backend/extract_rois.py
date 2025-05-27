import torch
import numpy as np
import pandas as pd
from pathlib import Path
import logging
import networkx as nx
import pickle
from models.gat import SpaFGAN

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define interaction logic
INTERACTIONS = {
    "T-cell entry site": {"CD31": "high", "CD4": "high"},
    "Inflammatory zone": {"CD11b": "high", "CD20": "high"},
    "Oxidative stress niche": {"CD11b": "high", "Catalase": "high"},
    "B-cell infiltration": {"CD20": "high", "CD31": "high"},
    "Dendritic signal": {"CD11c": "high"}
}

def normalize(value, marker, df):
    """Normalize based on data distribution"""
    marker_values = df[marker].values
    min_val = np.percentile(marker_values, 10)  # Use 10th percentile as min
    max_val = np.percentile(marker_values, 90)  # Use 90th percentile as max
    return max(0.0, min(1.0, (value - min_val) / (max_val - min_val)))

def detect_interactions(row, df, marker_name):
    interactions = []
    interaction_scores = []
    interaction_values = {
        "T-cell entry site": 0.0,
        "B-cell infiltration": 0.0,
        "Oxidative stress niche": 0.0
    }
    
    # T-cell entry site: CD31 + CD4
    if 'CD31' in df.columns and 'CD4' in df.columns:
        cd31_val = row.get('CD31', 0)
        cd4_val = row.get('CD4', 0)
        # MinMax normalization
        cd31_norm = normalize(cd31_val, 'CD31', df)
        cd4_norm = normalize(cd4_val, 'CD4', df)
        t_cell_score = (cd31_norm + cd4_norm) / 2
        interaction_values["T-cell entry site"] = t_cell_score
        if t_cell_score > 0.5:  # Threshold for T-cell entry site
            interactions.append("T-cell entry site")
            interaction_scores.append(t_cell_score)
    
    # B-cell infiltration: CD20 + CD31
    if 'CD20' in df.columns and 'CD31' in df.columns:
        cd20_val = row.get('CD20', 0)
        cd31_val = row.get('CD31', 0)
        # MinMax normalization
        cd20_norm = normalize(cd20_val, 'CD20', df)
        cd31_norm = normalize(cd31_val, 'CD31', df)
        b_cell_score = (cd20_norm + cd31_norm) / 2
        interaction_values["B-cell infiltration"] = b_cell_score
        if b_cell_score > 0.5:  # Threshold for B-cell infiltration
            interactions.append("B-cell infiltration")
            interaction_scores.append(b_cell_score)
    
    # Oxidative stress niche: CD11b + Catalase
    if 'CD11b' in df.columns and 'Catalase' in df.columns:
        cd11b_val = row.get('CD11b', 0)
        catalase_val = row.get('Catalase', 0)
        # MinMax normalization
        cd11b_norm = normalize(cd11b_val, 'CD11b', df)
        catalase_norm = normalize(catalase_val, 'Catalase', df)
        oxidative_score = (cd11b_norm + catalase_norm) / 2
        interaction_values["Oxidative stress niche"] = oxidative_score
        if oxidative_score > 0.3:  # Lower threshold for oxidative stress
            interactions.append("Oxidative stress niche")
            interaction_scores.append(oxidative_score)
    
    # Sort interactions by their scores in descending order
    if interactions:
        sorted_pairs = sorted(zip(interactions, interaction_scores), key=lambda x: x[1], reverse=True)
        interactions = [pair[0] for pair in sorted_pairs]
        interaction_scores = [pair[1] for pair in sorted_pairs]
    
    # Filter interactions based on marker
    if marker_name == "CD31":
        # For CD31, only show T-cell entry site and B-cell infiltration
        interaction_values = {
            "T-cell entry site": interaction_values["T-cell entry site"],
            "B-cell infiltration": interaction_values["B-cell infiltration"]
        }
    elif marker_name == "CD11b":
        # For CD11b, only show oxidative stress niche
        interaction_values = {
            "Oxidative stress niche": interaction_values["Oxidative stress niche"]
        }
    
    return interactions, interaction_scores, interaction_values

def process_marker(marker_name, model_path, feature_path, output_dir):
    """Process a single marker to extract ROIs"""
    logger.info(f"Processing ROI extraction for {marker_name}")
    
    # Load the trained model
    model = SpaFGAN(in_channels=len(INTERACTIONS))
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    # Load cell features
    df = pd.read_csv(feature_path)
    
    # Load spatial graph
    graph_path = Path(output_dir) / f"spatial_graph_{marker_name}.pkl"
    with open(graph_path, 'rb') as f:
        G = pickle.load(f)
    
    # Get ROI nodes
    roi_nodes = [n for n in G.nodes() if G.nodes[n].get('type') == 'roi']
    if not roi_nodes:
        logger.warning(f"No ROI nodes found for {marker_name}")
        return
    
    # Calculate interaction scores for each ROI
    roi_rows = []
    for node_id in roi_nodes:
        node_data = G.nodes[node_id]
        cell_data = df[df['cell_id'] == int(node_id)].iloc[0] if int(node_id) in df['cell_id'].values else None
        
        if cell_data is None:
            continue
            
        row = {
            "roi_id": f"ROI_{len(roi_rows)+1}",
            "cell_id": str(node_id),
            "x": float(cell_data['x']),
            "y": float(cell_data['y']),
            "z": float(cell_data['z']),
            "spafgan_score": float(node_data.get('spafgan_score', 0.0))
        }
        
        # Add marker values
        for marker in df.columns:
            if marker not in ['cell_id', 'x', 'y', 'z']:
                row[marker] = float(cell_data[marker])
        
        # Get interaction values
        _, _, interaction_values = detect_interactions(row, df, marker_name)
        
        # Add interaction values to row
        for interaction_name, value in interaction_values.items():
            row[interaction_name] = value
        
        roi_rows.append(row)
    
    # Sort ROI rows by spafgan_score in descending order
    roi_rows.sort(key=lambda x: x["spafgan_score"], reverse=True)
    
    # Save results
    output_df = pd.DataFrame(roi_rows)
    out_path = Path(output_dir) / f"extracted_rois_{marker_name}.csv"
    output_df.to_csv(out_path, index=False)
    logger.info(f"{len(output_df)} ROIs for {marker_name} saved to {out_path}")

def main():
    """Main function to process all markers"""
    backend_dir = Path(__file__).parent.resolve()
    output_dir = backend_dir / "output"
    
    for marker in ["CD31", "CD11b"]:
        model_path = output_dir / f"best_spafgan_model_{marker}.pt"
        feature_path = output_dir / f"cell_features_{marker}.csv"
        
        if not model_path.exists():
            logger.warning(f"Model file not found for {marker}, skipping...")
            continue
            
        if not feature_path.exists():
            logger.warning(f"Feature file not found for {marker}, skipping...")
            continue
        
        process_marker(marker, model_path, feature_path, output_dir)

if __name__ == "__main__":
    main()
