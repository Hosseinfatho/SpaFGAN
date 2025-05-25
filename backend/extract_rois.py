import torch
import numpy as np
import networkx as nx
import pandas as pd
from pathlib import Path
import pickle
import logging
from models.gat import SpaFGAN
from torch_geometric.utils import from_networkx

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define interaction logic
INTERACTIONS = {
    "T-cell entry site": {"CD31": "high", "CD4": "high"},
    "Inflammatory zone": {"CD11b": "high", "CD20": "high"},
    "Oxidative stress niche": {"CD11b": "high", "Catalase": "high"},
    "B-cell infiltration": {"CD20": "high", "CD31": "high"},
    "Dendritic signal": {"CD11c": "high"}  # 
}

# Marker thresholds
THRESHOLDS = {
    "CD31": 1000,     # ~5% of range: 1000/20000
    "CD20": 1500,     # ~20% of 7000
    "CD11b": 1200,    # ~20% of 6000
    "CD4": 3000,      # ~30% of 10000
    "CD11c": 300,     # ~30% of 1432
    "Catalase": 2000  # ~30% of 7000
}

def detect_interactions(row):
    interactions = []
    for label, rule in INTERACTIONS.items():
        match = all(row.get(marker, 0) > THRESHOLDS.get(marker, 0) for marker in rule)
        if match:
            interactions.append(label)
    return interactions

def process_marker(marker_name, model_path, graph_path, feature_path, output_dir):
    with open(graph_path, "rb") as f:
        G = pickle.load(f)

    df = pd.read_csv(feature_path).set_index("cell_id")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SpaFGAN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    for node in G.nodes:
        G.nodes[node]["x"] = torch.tensor(G.nodes[node]["features"], dtype=torch.float32)
        if node in df.index:
            for marker in THRESHOLDS:
                G.nodes[node][marker] = df.loc[node, marker] if marker in df.columns else 0

    data = from_networkx(G, group_node_attrs=["x"]).to(device)

    with torch.no_grad():
        scores = model(data.x, data.edge_index).cpu().numpy()

    threshold = 0.7 if marker_name == "CD31" else 0.2 if marker_name == "CD11b" else 0.1
    roi_node_indices = np.where(scores > threshold)[0]
    roi_nodes = [list(G.nodes)[i] for i in roi_node_indices]

    # Option B fallback if no ROI nodes found
    if not roi_nodes:
        logger.warning(f"⚠️ No ROI candidates found for {marker_name} at threshold {threshold}, applying Option B fallback")
        roi_nodes = list(G.nodes)

    subgraph = G.subgraph(roi_nodes)
    components = list(nx.connected_components(subgraph))

    logger.info(f"[{marker_name}] Found {len(components)} ROI components")

    roi_rows = []
    for i, comp in enumerate(components):
        for node_id in comp:
            node = G.nodes[node_id]
            x, y, z = node["pos"][2], node["pos"][1], node["pos"][0]
            row = {
                "roi_id": f"ROI_{i+1}",
                "cell_id": node_id,
                "x": x * 8,
                "y": y * 8,
                "z": z,
                "scores": np.atleast_1d(scores)  # Ensure it's at least 1D
            }
            for marker in THRESHOLDS:
                row[marker] = float(node.get(marker, 0))
            row["interactions"] = "; ".join(detect_interactions(row))
            roi_rows.append(row)

    output_df = pd.DataFrame(roi_rows)
    out_path = Path(output_dir) / f"roi_cells_{marker_name}.csv"
    output_df.to_csv(out_path, index=False)
    logger.info(f" Saved {len(output_df)} ROI cells for {marker_name} to {out_path}")

def main():
    backend_dir = Path(__file__).parent.resolve()
    output_dir = backend_dir / "output"
    model_path = output_dir / "spafgan_model.pt"

    for marker in ["CD31", "CD11b", "CD11c"]:
        graph_path = output_dir / f"spatial_graph_{marker}.pkl"
        feature_path = output_dir / f"cell_features_{marker}.csv"
        logger.info(f" Processing ROI extraction for {marker}")
        process_marker(marker, model_path, graph_path, feature_path, output_dir)

if __name__ == "__main__":
    main()
