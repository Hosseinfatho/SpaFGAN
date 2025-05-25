import os
import pickle
import logging
from pathlib import Path
import networkx as nx

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def analyze_graph(graph_path):
    """Load and analyze a spatial graph"""
    if not graph_path.exists():
        logger.error(f"Graph file not found at {graph_path}")
        return

    logger.info(f"Loading graph from: {graph_path}")
    with open(graph_path, "rb") as f:
        G = pickle.load(f)

    logger.info("Graph successfully loaded.")
    logger.info(f"Number of nodes: {G.number_of_nodes()}")
    logger.info(f"Number of edges: {G.number_of_edges()}")
    if G.number_of_nodes() > 0:
        avg_degree = 2 * G.number_of_edges() / G.number_of_nodes()
        logger.info(f"Average degree: {avg_degree:.2f}")
    else:
        logger.warning("Empty graph.")

def main():
    backend_dir = Path(__file__).parent.resolve()
    output_dir = backend_dir / "output"

    graph_files = list(output_dir.glob("spatial_graph_*.pkl"))
    if not graph_files:
        logger.error("❌ No spatial graph files found.")
        return

    for graph_file in graph_files:
        analyze_graph(graph_file)

if __name__ == "__main__":
    logger.info("🔍 Starting graph analysis...")
    main()
