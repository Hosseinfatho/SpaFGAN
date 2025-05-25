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

MARKERS = ["CD31", "CD20", "CD11b", "CD4", "CD11c", "Catalase"]

def build_spatial_graph(csv_path, marker_name, radius):
    """Build a spatial graph using all markers as features, based on a specific marker's CSV and radius"""
    try:
        df = pd.read_csv(csv_path)
        if df.empty:
            logger.warning(f"No cells found in {csv_path}, skipping.")
            return None

        logger.info(f"Loaded {len(df)} cells from {csv_path}")

        # Use all 6 markers as node features
        features = df[MARKERS].values
        coords = df[["z", "y", "x"]].values
        node_ids = df["cell_id"].values

        G = nx.Graph()

        for idx, nid in enumerate(node_ids):
            G.add_node(nid, features=features[idx], pos=coords[idx])

        nbrs = NearestNeighbors(radius=radius)
        nbrs.fit(coords)
        neighbors = nbrs.radius_neighbors(coords, return_distance=False)

        for i, indices in enumerate(neighbors):
            src_id = node_ids[i]
            for j in indices:
                tgt_id = node_ids[j]
                if src_id != tgt_id and not G.has_edge(src_id, tgt_id):
                    G.add_edge(src_id, tgt_id)

        logger.info(f"Graph for {marker_name}: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

        output_path = Path(csv_path).parent / f"spatial_graph_{marker_name}.pkl"
        with open(output_path, "wb") as f:
            pickle.dump(G, f)
        logger.info(f"Saved graph to {output_path}")

        return G

    except Exception as e:
        logger.error(f"Failed to build graph for {marker_name}: {e}", exc_info=True)
        return None

def main():
    backend_dir = Path(__file__).parent.resolve()
    output_dir = backend_dir / "output"
    marker_files = list(output_dir.glob("cell_features_*.csv"))

    if not marker_files:
        logger.error("No cell_features_*.csv files found in output directory.")
        return

    for file in marker_files:
        marker = file.stem.replace("cell_features_", "")
        radius = 30.0 if marker == "CD31" else 150.0
        logger.info(f" Processing marker: {marker} (radius={radius})")
        build_spatial_graph(file, marker, radius)

if __name__ == "__main__":
    logger.info(" Starting multi-marker spatial graph construction...")
    main()
