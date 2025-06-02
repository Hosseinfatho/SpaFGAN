import pandas as pd
import numpy as np
import networkx as nx
from sklearn.neighbors import NearestNeighbors
import logging
import os
from pathlib import Path
import pickle

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define marker groups
PRIMARY_MARKERS = ["CD31", "CD11b", "CD11c"]  # ROI centers
SECONDARY_MARKERS = ["CD4", "CD20", "Catalase"]  # Cells to connect
ALL_MARKERS = PRIMARY_MARKERS + SECONDARY_MARKERS

def calculate_edge_weight(cell1_type, cell2_type, distance):
    """Calculate edge weight based on cell types and distance"""
    # Base weight from distance (closer = higher weight)
    distance_weight = 1.0 / (1.0 + distance)
    
    # Type weights (higher for connections between different types)
    type_weights = {
        ("CD31", "CD11b"): 1.5,  # Strong connection between primary markers
        ("CD31", "CD11c"): 1.4,  # Strong connection with CD11c
        ("CD31", "CD4"): 1.2,    # Moderate connection with CD4
        ("CD31", "CD20"): 1.2,   # Moderate connection with CD20
        ("CD31", "Catalase"): 1.1,  # Weak connection with Catalase
        ("CD11b", "CD11c"): 1.5,  # Strong connection between immune markers
        ("CD11b", "CD4"): 1.2,
        ("CD11b", "CD20"): 1.2,
        ("CD11b", "Catalase"): 1.1,
        ("CD11c", "CD4"): 1.3,    # Moderate-strong connection
        ("CD11c", "CD20"): 1.3,   # Moderate-strong connection
        ("CD11c", "Catalase"): 1.2,  # Moderate connection
    }
    
    # Get type weight (default to 1.0 if not specified)
    type_key = tuple(sorted([cell1_type, cell2_type]))
    type_weight = type_weights.get(type_key, 1.0)
    
    return distance_weight * type_weight

def build_spatial_graph(marker, radius=50.0):
    """Build a spatial graph using ROIs as centers and connecting nearby cells"""
    logger.info(f"Processing marker: {marker} (radius={radius})")
    
    # Load ROI and cell data
    backend_dir = Path(__file__).parent.resolve()
    output_dir = backend_dir / "output"
    roi_file = output_dir / f"roi_cells_{marker}.csv"
    cell_file = output_dir / f"cell_features_{marker}.csv"
    
    if not roi_file.exists() or not cell_file.exists():
        logger.error(f"Required files not found for {marker}")
        return None
    
    # Load and clean data
    roi_data = pd.read_csv(roi_file)
    cell_data = pd.read_csv(cell_file)
    
    # Convert cell_id to clean string format (no decimal, no prefix)
    roi_data['cell_id'] = roi_data['cell_id'].apply(lambda x: str(int(float(x))))
    cell_data['cell_id'] = cell_data['cell_id'].apply(lambda x: str(int(float(x))))
    
    logger.info(f"Loaded {len(roi_data)} ROIs and {len(cell_data)} cells")

    # Create graph
    G = nx.Graph()

    # Add ROI nodes with all marker values
    for _, row in roi_data.iterrows():
        roi_id = str(int(float(row['cell_id'])))
        if roi_id in cell_data['cell_id'].values and roi_id not in G.nodes():
            cell_row = cell_data[cell_data['cell_id'] == roi_id].iloc[0]
            pos = [float(cell_row['x']), float(cell_row['y']), float(cell_row['z'])]
            G.add_node(roi_id, 
                      type="roi",
                      pos=pos,
                      marker=marker,
                      spafgan_score=float(row.get('spafgan_score', 1.0)))  # Use actual score if available
            for marker_name in ALL_MARKERS:
                if marker_name in cell_row:
                    G.nodes[roi_id][marker_name] = float(cell_row[marker_name])
                else:
                    G.nodes[roi_id][marker_name] = 0.0

    # Add cell nodes with all marker values
    for _, row in cell_data.iterrows():
        cell_id = str(int(float(row['cell_id'])))
        if cell_id not in G.nodes():
            pos = [float(row['x']), float(row['y']), float(row['z'])]
            G.add_node(cell_id,
                      type="cell",
                      pos=pos,
                      marker=marker)
            for marker_name in ALL_MARKERS:
                if marker_name in row:
                    G.nodes[cell_id][marker_name] = float(row[marker_name])
                else:
                    G.nodes[cell_id][marker_name] = 0.0

    # Connect nodes within radius
    roi_coords = np.array([G.nodes[n]['pos'] 
                          for n in G.nodes() if G.nodes[n]['type'] == 'roi'])
    cell_coords = np.array([G.nodes[n]['pos'] 
                           for n in G.nodes() if G.nodes[n]['type'] == 'cell'])
    
    if len(roi_coords) > 0 and len(cell_coords) > 0:
        nn = NearestNeighbors(n_neighbors=1, radius=radius)
        nn.fit(roi_coords)
        distances, indices = nn.kneighbors(cell_coords)
        
        for i, (dist, idx) in enumerate(zip(distances, indices)):
            if dist[0] <= radius:
                cell_node = list(G.nodes())[i + len(roi_data)]
                roi_node = list(G.nodes())[idx[0]]
                weight = calculate_edge_weight(marker, marker, dist[0])
                G.add_edge(cell_node, roi_node, weight=weight)
    
    # Connect nearby cells
    if len(cell_coords) > 1:
        # Adjust number of neighbors based on available cells
        n_neighbors = min(5, len(cell_coords) - 1)
        nn = NearestNeighbors(n_neighbors=n_neighbors, radius=radius)
        nn.fit(cell_coords)
        distances, indices = nn.kneighbors(cell_coords)
        
        for i, (dists, idxs) in enumerate(zip(distances, indices)):
            for j, (dist, idx) in enumerate(zip(dists, idxs)):
                if i != idx and dist <= radius:
                    cell1 = list(G.nodes())[i]
                    cell2 = list(G.nodes())[idx]
                    weight = calculate_edge_weight(marker, marker, dist)
                    G.add_edge(cell1, cell2, weight=weight)
    
    # Remove isolated nodes
    isolated_nodes = list(nx.isolates(G))
    G.remove_nodes_from(isolated_nodes)
    
    if len(G.nodes()) > 0:
        logger.info(f"Created graph with {len(G.nodes())} nodes and {len(G.edges())} edges")
        logger.info(f"Average degree: {2*len(G.edges())/len(G.nodes()):.2f}")
        
        # Save graph
        output_file = output_dir / f"spatial_graph_{marker}.pkl"
        with open(str(output_file), 'wb') as f:
            pickle.dump(G, f)
        logger.info(f"Graph saved to {output_file}")
        return G
    else:
        logger.error(f"Failed to create graph for {marker}")
        return None

def main():
    """Main function to build spatial graphs"""
    logger.info("Starting ROI-based spatial graph construction...")
    
    # Build graph for each primary marker
    for marker in PRIMARY_MARKERS:
        radius = 50.0 if marker in ["CD11b", "CD11c"] else 50.0  # Larger radius for immune markers
        build_spatial_graph(marker, radius)

if __name__ == "__main__":
    main()
