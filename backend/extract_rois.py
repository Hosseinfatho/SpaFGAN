import torch
import numpy as np
import networkx as nx
import pandas as pd
from pathlib import Path
import pickle
import logging
from models.gat import SpaFGAN
from torch_geometric.utils import from_networkx

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_model_and_data(model_path="backend/output/spafgan_model.pt", 
                       graph_path="backend/output/spatial_graph.pkl"):
    """
    Load the trained model and original graph.
    
    Args:
        model_path (str): Path to the saved model
        graph_path (str): Path to the original graph
        
    Returns:
        tuple: (model, graph, data)
    """
    try:
        # Load original graph
        logger.info(f"Loading original graph from {graph_path}")
        with open(graph_path, "rb") as f:
            G = pickle.load(f)
        
        # Load model
        logger.info(f"Loading model from {model_path}")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = SpaFGAN().to(device)
        model.load_state_dict(torch.load(model_path, weights_only=True))
        model.eval()
        
        # Prepare data for prediction
        for n in G.nodes:
            G.nodes[n]["x"] = torch.tensor(G.nodes[n]["features"], dtype=torch.float32)
        
        data = from_networkx(G, group_node_attrs=["x"])
        data = data.to(device)
        
        return model, G, data
        
    except Exception as e:
        logger.error(f"Error loading model and data: {str(e)}")
        raise

def extract_rois(model, G, data, threshold=0.7):
    """
    Extract ROIs from model predictions.
    
    Args:
        model (SpaFGAN): Trained model
        G (nx.Graph): Original graph
        data (torch_geometric.data.Data): Graph data
        threshold (float): Score threshold for ROI selection
        
    Returns:
        tuple: (scores, roi_node_indices, components)
    """
    try:
        # Get predictions
        logger.info("Computing ROI scores")
        with torch.no_grad():
            scores = model(data.x, data.edge_index).cpu().numpy()
        
        # Threshold scores
        logger.info(f"Thresholding scores at {threshold}")
        roi_node_indices = np.where(scores > threshold)[0]
        logger.info(f"Selected {len(roi_node_indices)} candidate ROI cells")
        
        # Extract connected components
        logger.info("Extracting connected components")
        roi_nodes = [list(G.nodes)[i] for i in roi_node_indices]
        subgraph = G.subgraph(roi_nodes).copy()
        components = list(nx.connected_components(subgraph))
        logger.info(f"Extracted {len(components)} distinct ROI subgraphs")
        
        return scores, roi_node_indices, components
        
    except Exception as e:
        logger.error(f"Error extracting ROIs: {str(e)}")
        raise

def save_roi_info(G, components, scores, output_dir="backend/output"):
    """
    Save ROI information to CSV.
    
    Args:
        G (nx.Graph): Original graph
        components (list): List of ROI components
        scores (np.ndarray): ROI scores
        output_dir (str): Output directory
    """
    try:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("Saving ROI information")
        roi_info = []
        
        for i, comp in enumerate(components):
            for node_id in comp:
                node = G.nodes[node_id]
                roi_info.append({
                    "roi_id": f"ROI_{i+1}",
                    "cell_id": node_id,
                    "x": node["pos"][2],
                    "y": node["pos"][1],
                    "z": node["pos"][0],
                    "score": float(scores[list(G.nodes).index(node_id)])
                })
        
        df_roi = pd.DataFrame(roi_info)
        output_path = output_dir / "roi_cells.csv"
        df_roi.to_csv(output_path, index=False)
        logger.info(f"Saved ROI cells to {output_path}")
        
    except Exception as e:
        logger.error(f"Error saving ROI information: {str(e)}")
        raise

def main():
    try:
        # Load model and data
        model, G, data = load_model_and_data()
        
        # Extract ROIs
        scores, roi_node_indices, components = extract_rois(model, G, data)
        
        # Save ROI information
        save_roi_info(G, components, scores)
        
        logger.info("ROI extraction completed successfully")
        
    except Exception as e:
        logger.error(f"ROI extraction failed: {str(e)}")
        raise

if __name__ == "__main__":
    main() 