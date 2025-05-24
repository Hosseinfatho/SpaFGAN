import torch
import numpy as np
import networkx as nx
import pandas as pd
from pathlib import Path
import pickle
import logging
import json
from models.gat import SpaFGAN
from torch_geometric.utils import from_networkx
from shapely.geometry import mapping, Polygon, MultiPoint
import alphashape

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

INTERACTIONS = {
    "T-cell entry site": {"CD31": "high", "CD4": "high"},
    "Inflammatory zone": {"CD11b": "high", "CD20": "high"},
    "Oxidative stress niche": {"CD11b": "high", "Catalase": "high"},
    "B-cell infiltration": {"CD20": "high", "CD31": "high"},
}

THRESHOLDS = {
    "CD31": 1000,
    "CD4": 500,
    "CD11b": 700,
    "CD20": 600,
    "Catalase": 800,
}

def detect_interactions(marker_means):
    labels = []
    for name, rule in INTERACTIONS.items():
        match = True
        for marker, level in rule.items():
            if level == "high" and marker in marker_means:
                if marker_means[marker] <= THRESHOLDS.get(marker, 500):
                    match = False
                    break
            else:
                match = False
                break
        if match:
            labels.append(name)
    return labels

def load_model_and_data(model_path="backend/output/spafgan_model.pt", 
                        graph_path="backend/output/spatial_graph.pkl",
                        feature_path="backend/output/cell_features.csv"):
    logger.info(f"Loading original graph from {graph_path}")
    with open(graph_path, "rb") as f:
        G = pickle.load(f)

    logger.info(f"Loading marker data from {feature_path}")
    df_cells = pd.read_csv(feature_path).set_index("cell_id")

    logger.info(f"Loading model from {model_path}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SpaFGAN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    for n in G.nodes:
        G.nodes[n]["x"] = torch.tensor(G.nodes[n]["features"], dtype=torch.float32)
        if n in df_cells.index:
            for marker in THRESHOLDS:
                if marker in df_cells.columns:
                    G.nodes[n][marker] = df_cells.loc[n, marker]

    data = from_networkx(G, group_node_attrs=["x"])
    data = data.to(device)

    return model, G, data, df_cells

def extract_rois(model, G, data, threshold=0.7):
    logger.info("Computing ROI scores")
    with torch.no_grad():
        scores = model(data.x, data.edge_index).cpu().numpy()

    logger.info(f"Thresholding scores at {threshold}")
    roi_node_indices = np.where(scores > threshold)[0]
    logger.info(f"Selected {len(roi_node_indices)} candidate ROI cells")

    logger.info("Extracting connected components")
    roi_nodes = [list(G.nodes)[i] for i in roi_node_indices]
    subgraph = G.subgraph(roi_nodes).copy()
    components = list(nx.connected_components(subgraph))
    logger.info(f"Extracted {len(components)} distinct ROI subgraphs")

    return scores, roi_node_indices, components

def save_roi_cells(G, components, scores, df_cells, output_dir="backend/output"):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Saving ROI cells with markers")
    roi_cell_rows = []

    for i, comp in enumerate(components):
        for node_id in comp:
            node = G.nodes[node_id]
            x, y, z = node["pos"][2], node["pos"][1], node["pos"][0]
            scaledX = x * 8 if x is not None else None
            scaledY = y * 8 if y is not None else None
            row = {
                "roi_id": f"ROI_{i+1}",
                "cell_id": node_id,
                "x": scaledX,
                "y": scaledY,
                "z": z,
                "score": float(scores[list(G.nodes).index(node_id)])
            }
            for marker in THRESHOLDS:
                if marker in node:
                    row[marker] = float(node[marker])
            roi_cell_rows.append(row)

    df_roi_cells = pd.DataFrame(roi_cell_rows)
    output_path = output_dir / "roi_cells.csv"
    df_roi_cells.to_csv(output_path, index=False)
    logger.info(f"✅ Saved ROI cells to {output_path}")

def main():
    try:
        model, G, data, df_cells = load_model_and_data()
        scores, roi_node_indices, components = extract_rois(model, G, data)
        save_roi_cells(G, components, scores, df_cells)
        logger.info("✅ ROI cell file with markers created successfully")
    except Exception as e:
        logger.error(f"❌ ROI extraction failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()
