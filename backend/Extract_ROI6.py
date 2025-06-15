import torch
import torch.nn as nn
from torch_geometric.data import Data
import numpy as np
import json
import os
from pathlib import Path
import logging
from train_gat4 import MarkerGAT, INTERACTIONS, MARKERS

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_graphs(marker):
    """Load graphs for a specific marker and ensure all markers are included"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, "output", f"roi_graphs_{marker}.pt")
    if not Path(file_path).exists():
        logger.error(f"Graph file not found: {file_path}")
        return None
    
    data = torch.load(file_path)
    graphs = []
    
    # Convert dictionary graphs to PyTorch Geometric Data objects
    for graph_dict in data['graphs']:
        graph = graph_dict['graph']
        if isinstance(graph, dict):
            # Create full feature tensor with all markers
            x = torch.zeros((graph['x'].size(0), len(MARKERS)), device=graph['x'].device)
            
            # Get the marker index
            marker_idx = MARKERS.index(marker)
            
            # Copy the original features to the correct position
            x[:, marker_idx] = graph['x'].squeeze()
            
            # Convert dictionary to Data object with full features
            graph = Data(
                x=x,
                edge_index=graph['edge_index'],
                edge_attr=graph['edge_attr'],
                pos=graph['pos'],
                marker=marker
            )
        graphs.append(graph)
    
    return graphs

def compute_interaction_score(graph, interaction_name):
    """Compute interaction score for a specific interaction"""
    markers = INTERACTIONS[interaction_name]
    score = 0.0
    
    for marker, level in markers.items():
        if marker in MARKERS:
            marker_idx = MARKERS.index(marker)
            if marker_idx < graph.x.size(1):
                marker_expr = graph.x[:, marker_idx].mean().item()
                if level == "high":
                    score += marker_expr
    
    return score / len(markers)

def extract_embeddings(model, graphs, device, interaction_name):
    """Extract embeddings for all ROIs"""
    model.eval()
    embeddings = []
    scores = []
    
    with torch.no_grad():
        for graph in graphs:
            if isinstance(graph, dict):
                graph = Data(
                    x=graph['graph']['x'],
                    edge_index=graph['graph']['edge_index'],
                    edge_attr=graph['graph']['edge_attr'],
                    pos=graph['graph']['pos']
                )
            
            graph = graph.to(device)
            emb = model(graph.x, graph.edge_index, graph.edge_attr)
            score = compute_interaction_score(graph, interaction_name)
            
            embeddings.append(emb.cpu().numpy())
            scores.append(score)
    
    return embeddings, np.array(scores)

def calculate_distance(roi1, roi2):
    """Calculate Euclidean distance between two ROIs"""
    return ((roi1['x'] - roi2['x'])**2 + (roi1['y'] - roi2['y'])**2)**0.5

def save_interaction_results(interaction_name, graphs, scores, output_dir):
    """Save results for a specific interaction"""
    # Create summary of all ROIs
    roi_summary = []
    for i, (graph, score) in enumerate(zip(graphs, scores)):
        if isinstance(graph, dict):
            pos = graph['graph']['pos']
        else:
            pos = graph.pos
        
        # Debug print for position tensor
        logger.debug(f"Position tensor shape: {pos.shape}, type: {type(pos)}")
        
        try:
            # Handle different tensor shapes
            if len(pos.shape) == 1:
                x, y = pos[0].item(), pos[1].item()
            else:
                x, y = pos[0, 0].item(), pos[0, 1].item()
            
            roi_data = {
                "x": float(x),
                "y": float(y),
                "score": float(score),
                "interaction": interaction_name
            }
            
            # Add z coordinate if available
            if len(pos.shape) == 1 and pos.size(0) > 2:
                roi_data["z"] = float(pos[2].item())
            elif len(pos.shape) > 1 and pos.size(1) > 2:
                roi_data["z"] = float(pos[0, 2].item())
            
            roi_summary.append(roi_data)
            
        except Exception as e:
            logger.error(f"Error processing position for ROI {i}: {str(e)}")
            logger.error(f"Position tensor: {pos}")
            continue
    
    if not roi_summary:
        logger.error(f"No valid ROIs found for {interaction_name}")
        return
    
    # Sort ROIs by score in descending order
    roi_summary.sort(key=lambda x: x["score"], reverse=True)
    
    # Select top ROIs with minimum distance of 50 pixels
    selected_rois = []
    min_distance = 50  # minimum distance between ROIs in pixels
    
    for roi in roi_summary:
        # Check if this ROI is far enough from all previously selected ROIs
        if all(calculate_distance(roi, selected_roi) > min_distance for selected_roi in selected_rois):
            selected_rois.append(roi)
            if len(selected_rois) >= 10:  # Stop after selecting 10 ROIs
                break
    
    logger.info(f"\nSelected ROIs for {interaction_name} (minimum distance: {min_distance} pixels):")
    logger.info("-" * 50)
    for i, roi in enumerate(selected_rois, 1):
        logger.info(f"ROI {i}: Score = {roi['score']:.4f}, Position = ({roi['x']:.1f}, {roi['y']:.1f})")
    
    # Save to JSON file
    output_file = os.path.join(output_dir, f"extraction_roi_{interaction_name}.json")
    with open(output_file, 'w') as f:
        json.dump({
            "interaction_name": interaction_name,
            "rois": selected_rois,
            "total_rois": len(roi_summary),
            "selected_rois": len(selected_rois),
            "min_distance": min_distance,
            "score_range": {
                "max": float(selected_rois[0]['score']) if selected_rois else 0,
                "min": float(selected_rois[-1]['score']) if selected_rois else 0
            }
        }, f, indent=2)
    
    logger.info(f"\nSaved {len(selected_rois)} ROIs for {interaction_name} to {output_file}")

def main():
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Create output directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, "output")
    os.makedirs(output_dir, exist_ok=True)
    
    # Process each interaction
    for interaction_name in INTERACTIONS.keys():
        logger.info(f"\nProcessing interaction: {interaction_name}")
        
        # Load required graphs
        required_markers = list(INTERACTIONS[interaction_name].keys())
        graphs = []
        for marker in required_markers:
            marker_graphs = load_graphs(marker)
            if marker_graphs:
                graphs.extend(marker_graphs)
        
        if not graphs:
            logger.error(f"No graphs found for {interaction_name}")
            continue
        
        # Load model
        model = MarkerGAT(
            in_channels=len(MARKERS),
            hidden_channels=64,
            out_channels=32
        ).to(device)
        
        # Load best model checkpoint
        checkpoint_path = os.path.join(script_dir, "output", f"best_model_{required_markers[0]}.pt")
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path)
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            logger.error(f"Model checkpoint not found: {checkpoint_path}")
            continue
        
        # Extract embeddings and compute scores
        embeddings, scores = extract_embeddings(model, graphs, device, interaction_name)
        
        # Save results
        save_interaction_results(interaction_name, graphs, scores, output_dir)

if __name__ == "__main__":
    main() 