import pandas as pd
import numpy as np
import networkx as nx
from sklearn.neighbors import NearestNeighbors
import logging
from pathlib import Path
import pickle

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def build_spatial_graph(csv_path, radius=30.0):
    """
    Build a spatial graph from cell features where:
    - Nodes are cells with biomarker intensities as features
    - Edges connect cells within specified radius
    """
    try:
        # Load cell features
        logger.info(f"Loading cell features from {csv_path}")
        df = pd.read_csv(csv_path)
        logger.info(f"Loaded {len(df)} cells")

        # Node features: biomarker intensities
        features = df[["CD31", "CD20", "CD11b", "CD4", "CD11c", "Catalase"]].values

        # Node positions
        coords = df[["z", "y", "x"]].values
        node_ids = df["cell_id"].values

        # Build graph
        logger.info("Building graph...")
        G = nx.Graph()

        # Add nodes with features and positions
        for idx, nid in enumerate(node_ids):
            G.add_node(nid, features=features[idx], pos=coords[idx])

        # Radius-based neighbor search
        logger.info(f"Finding neighbors within radius {radius}")
        nbrs = NearestNeighbors(radius=radius)
        nbrs.fit(coords)
        neighbors = nbrs.radius_neighbors(coords, return_distance=False)

        # Add edges between neighbors
        edge_count = 0
        for i, indices in enumerate(neighbors):
            src_id = node_ids[i]
            for j in indices:
                tgt_id = node_ids[j]
                if src_id != tgt_id and not G.has_edge(src_id, tgt_id):
                    G.add_edge(src_id, tgt_id)
                    edge_count += 1

        logger.info(f"Graph built with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
        
        # Save graph in pickle format
        output_path = Path(csv_path).parent / "spatial_graph.pkl"
        with open(output_path, "wb") as f:
            pickle.dump(G, f)
        logger.info(f"Graph saved to {output_path}")

        # Print graph statistics
        logger.info("\nGraph Statistics:")
        logger.info(f"Number of nodes: {G.number_of_nodes()}")
        logger.info(f"Number of edges: {G.number_of_edges()}")
        logger.info(f"Average degree: {2 * G.number_of_edges() / G.number_of_nodes():.2f}")
        
        # Calculate and print feature statistics
        feature_means = np.mean(features, axis=0)
        feature_stds = np.std(features, axis=0)
        logger.info("\nFeature Statistics:")
        for i, name in enumerate(["CD31", "CD20", "CD11b", "CD4", "CD11c", "Catalase"]):
            logger.info(f"{name}: mean={feature_means[i]:.2f}, std={feature_stds[i]:.2f}")

        return G

    except Exception as e:
        logger.error(f"Error building spatial graph: {e}", exc_info=True)
        return None

def main():
    # Get the absolute path to the backend directory
    backend_dir = Path(__file__).parent.resolve()
    
    # Define input path
    csv_path = backend_dir / "output" / "cell_features.csv"
    
    logger.info(f"Input CSV path: {csv_path}")
    
    # Build graph
    G = build_spatial_graph(csv_path, radius=30.0)
    
    if G is not None:
        logger.info("✅ Graph construction completed successfully")
    else:
        logger.error("❌ Graph construction failed")

if __name__ == "__main__":
    logger.info("Starting spatial graph construction...")
    main() 