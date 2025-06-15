import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.data import Data
import logging
from pathlib import Path
from tqdm import tqdm

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define markers
MARKERS = ["CD31", "CD20", "CD11b", "CD4", "CD11c", "Catalase"]

# Define radius for ROI
RADIUS = 30.0

class MarkerGAT(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=4):
        super(MarkerGAT, self).__init__()
        self.gat1 = GATConv(in_channels, hidden_channels, heads=heads)
        self.gat2 = GATConv(hidden_channels * heads, out_channels, heads=1)
        
    def forward(self, x, edge_index, edge_attr):
        # First GAT layer with edge attributes
        x = F.elu(self.gat1(x, edge_index, edge_attr))
        # Second GAT layer
        x = F.elu(self.gat2(x, edge_index))
        return x

def find_neighbors_in_radius(roi_row, all_cells_df, radius=RADIUS):
    """Find all cells within radius of the ROI cell"""
    x, y = roi_row['x'], roi_row['y']
    marker = roi_row['marker']
    
    # Find cells within radius
    mask = (
        (all_cells_df['x'] >= x - radius) & 
        (all_cells_df['x'] <= x + radius) & 
        (all_cells_df['y'] >= y - radius) & 
        (all_cells_df['y'] <= y + radius)
    )
    
    # Calculate actual distances
    neighbors = all_cells_df[mask].copy()
    if len(neighbors) > 0:
        neighbors['distance'] = np.sqrt(
            (neighbors['x'] - x)**2 + 
            (neighbors['y'] - y)**2
        )
        # Filter by actual distance
        neighbors = neighbors[neighbors['distance'] <= radius]
    
    return neighbors

def build_roi_graph(roi_row, neighbors_df):
    """Build graph for a single ROI"""
    if len(neighbors_df) == 0:
        return None
    
    # Create node features
    node_features = []
    for marker in MARKERS:
        if marker in neighbors_df.columns:
            node_features.append(neighbors_df[marker].values)
    
    # Stack features
    x = torch.tensor(np.column_stack(node_features), dtype=torch.float32)
    
    # Create edge index and weights
    edge_index = []
    edge_weights = []
    
    # Add edges between all nodes
    for i in range(len(neighbors_df)):
        for j in range(i + 1, len(neighbors_df)):
            # Calculate distance weight
            dist = neighbors_df.iloc[i]['distance']
            dist_weight = 1.0 / (1.0 + dist)
            
            # Calculate marker compatibility
            m1 = neighbors_df.iloc[i]['marker']
            m2 = neighbors_df.iloc[j]['marker']
            
            # Add bidirectional edges
            edge_index.append([i, j])
            edge_index.append([j, i])
            edge_weights.append(dist_weight)
            edge_weights.append(dist_weight)
    
    if not edge_index:
        return None
    
    edge_index = torch.tensor(edge_index, dtype=torch.long).T
    edge_attr = torch.tensor(edge_weights, dtype=torch.float32).unsqueeze(1)
    
    # Create graph data
    graph = Data(
        x=x,  # Node features
        edge_index=edge_index,  # Edge connections
        edge_attr=edge_attr,  # Edge weights
        pos=torch.tensor(neighbors_df[['x', 'y']].values, dtype=torch.float32),  # Node positions
        marker=neighbors_df['marker'].values  # Node markers
    )
    
    return graph

def process_rois():
    """Process all ROIs and build their graphs"""
    # Load normalized cell features
    input_path = "output/Norm_all_cell_features.csv"
    if not Path(input_path).exists():
        logger.error(f"Input file not found: {input_path}")
        return
    
    df = pd.read_csv(input_path)
    logger.info(f"Loaded {len(df)} cells from {input_path}")
    
    # Initialize GAT models for each marker
    marker_models = {
        marker: MarkerGAT(
            in_channels=len(MARKERS),
            hidden_channels=64,
            out_channels=32
        ) for marker in MARKERS
    }
    
    # Process each ROI by marker
    marker_graphs = {marker: [] for marker in MARKERS}
    
    for marker in MARKERS:
        logger.info(f"\nProcessing {marker} ROIs...")
        marker_cells = df[df['marker'] == marker]
        
        for idx, row in tqdm(marker_cells.iterrows(), total=len(marker_cells), desc=f"Processing {marker}"):
            # Find neighbors
            neighbors = find_neighbors_in_radius(row, df)
            
            if len(neighbors) > 1:  # Need at least 2 nodes for a graph
                # Build graph
                graph = build_roi_graph(row, neighbors)
                if graph is not None:
                    # Log graph statistics
                    logger.info(f"\nGraph for {marker} ROI {idx}:")
                    logger.info(f"Number of nodes: {graph.num_nodes}")
                    logger.info(f"Number of edges: {graph.num_edges}")
                    logger.info(f"Node features shape: {graph.x.shape}")
                    logger.info(f"Edge attributes shape: {graph.edge_attr.shape}")
                    
                    marker_graphs[marker].append({
                        'roi_id': idx,
                        'graph': graph
                    })
    
    # Save results for each marker
    for marker in MARKERS:
        if marker_graphs[marker]:
            output_path = f"output/roi_graphs_{marker}.pt"
            torch.save({
                'graphs': marker_graphs[marker],
                'model_state': marker_models[marker].state_dict()
            }, output_path)
            logger.info(f"Saved {len(marker_graphs[marker])} {marker} ROI graphs to {output_path}")
    
    # Log statistics
    logger.info("\nROI Graph Statistics by Marker:")
    for marker in MARKERS:
        logger.info(f"{marker}: {len(marker_graphs[marker])} graphs")

if __name__ == "__main__":
    process_rois() 