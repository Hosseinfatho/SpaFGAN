import torch
import numpy as np
import pandas as pd
from pathlib import Path
import logging
from models.gat import SpaFGAN
import time
from sklearn.preprocessing import MinMaxScaler
from torch_geometric.utils import dense_to_sparse

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

def compute_interaction_score(row, df):
    # Get all interaction scores
    _, scores, _ = detect_interactions(row, df)
    return round(np.mean(scores), 3) if scores else 0.0

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

def create_spatial_graph(features, positions, radius=50):
    """
    Create a spatial graph based on radius-based neighborhood with weighted edges
    
    Args:
        features (torch.Tensor): Node features
        positions (torch.Tensor): Node positions (x, y, z)
        radius (float): Maximum distance for connecting nodes
        
    Returns:
        tuple: (edge_index, edge_weights)
    """
    # Calculate pairwise distances
    dist = torch.cdist(positions, positions)
    
    # Create adjacency matrix based on radius
    adj = (dist <= radius).float()
    
    # Remove self-loops
    adj.fill_diagonal_(0)
    
    # Calculate edge weights based on distance
    edge_weights = torch.exp(-dist / radius) * adj
    
    # Convert to edge index and weights
    edge_index = dense_to_sparse(adj)[0]
    edge_weights = edge_weights[edge_index[0], edge_index[1]]
    
    return edge_index, edge_weights

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def process_marker(marker_name, model_path, feature_path, output_dir):
    start_time = time.time()
    logger.info(f"========== Start processing marker: {marker_name} ==========")
    logger.info(f"Loading features from: {feature_path}")
    df = pd.read_csv(feature_path)
    logger.info(f"Total number of cells: {len(df)}")
    logger.info(f"Available columns: {list(df.columns)}")
    
    # Normalize input features
    feature_columns = ["CD31", "CD20", "CD11b", "CD4", "Catalase"]
    scaler = MinMaxScaler()
    normalized_features = scaler.fit_transform(df[feature_columns])
    
    # Prepare features and positions
    features = torch.tensor(normalized_features, dtype=torch.float32)
    positions = torch.tensor(df[["x", "y", "z"]].values, dtype=torch.float32)
    logger.info(f"Features shape: {features.shape}, Positions shape: {positions.shape}")
    
    # Print feature statistics after normalization
    logger.info(f"Normalized feature statistics for {marker_name}:")
    for i, col in enumerate(feature_columns):
        logger.info(f"{col}: Min={normalized_features[:, i].min():.3f}, Max={normalized_features[:, i].max():.3f}, Mean={normalized_features[:, i].mean():.3f}")
    
    logger.info("Building spatial graph based on radius-based neighborhood...")
    t0 = time.time()
    edge_index, edge_weights = create_spatial_graph(features, positions, radius=50)
    logger.info(f"Graph built. Number of edges: {edge_index.shape[1]}")
    logger.info(f"Graph construction time: {time.time() - t0:.2f} seconds")
    
    # Load model
    logger.info(f"Loading model from: {model_path}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SpaFGAN(
        in_channels=12,  # Updated to match saved model
        hidden_channels=32,
        out_channels=1,
        heads=2
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    logger.info("Model loaded successfully.")
    
    # Move data to device
    # Pad features to match model input dimension
    padded_features = torch.zeros((features.shape[0], 12), dtype=torch.float32, device=device)
    padded_features[:, :5] = features  # Copy original features
    features = padded_features
    edge_index = edge_index.to(device)
    edge_weights = edge_weights.to(device)
    
    logger.info("Starting model prediction...")
    t1 = time.time()
    with torch.no_grad():
        scores = model(features, edge_index, edge_weights).cpu().numpy()
    logger.info(f"Model prediction completed. Prediction time: {time.time() - t1:.2f} seconds")
    scores = np.atleast_1d(scores)
    
    # Normalize scores using MinMaxScaler
    scaler = MinMaxScaler()
    scores = scaler.fit_transform(scores.reshape(-1, 1)).flatten()
    
    # Print raw scores statistics
    logger.info(f"Raw score statistics for {marker_name}:")
    logger.info(f"Min: {scores.min():.3f} | Max: {scores.max():.3f} | Mean: {scores.mean():.3f} | Std: {scores.std():.3f}")
    
    # Apply threshold based on data distribution
    threshold = np.percentile(scores, 75)  # Use 75th percentile as threshold
    logger.info(f"Applying threshold: {threshold:.3f}")
    roi_indices = np.where(scores > threshold)[0]
    logger.info(f"Number of extracted ROI cells: {len(roi_indices)}")
    if len(roi_indices) == 0:
        logger.warning(f"No ROI cells found for {marker_name}.")
        logger.info(f"Total processing time: {time.time() - start_time:.2f} seconds")
        return
    
    # Create ROI dataframe with interaction values
    roi_rows = []
    for idx in roi_indices:
        row = {
            "roi_id": f"ROI_{len(roi_rows)+1}",
            "cell_id": str(int(df.iloc[idx]["cell_id"])),
            "x": float(df.iloc[idx]["x"]),
            "y": float(df.iloc[idx]["y"]),
            "z": float(df.iloc[idx]["z"]),
            "spafgan_score": float(scores[idx])
        }
        
        # Add marker values
        for marker in feature_columns:
            row[marker] = float(df.iloc[idx][marker])
        
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
    out_path = Path(output_dir) / f"roi_cells_{marker_name}.csv"
    output_df.to_csv(out_path, index=False)
    logger.info(f"{len(output_df)} ROI cells for {marker_name} saved to {out_path}.")
    logger.info(f"Total processing time: {time.time() - start_time:.2f} seconds")
    logger.info(f"========== End processing marker: {marker_name} ==========")

def main():
    backend_dir = Path(__file__).parent.resolve()
    output_dir = backend_dir / "output"
    
    for marker in ["CD31", "CD11b", "CD11c"]:
        model_path = output_dir / f"best_spafgan_model_{marker}.pt"
        feature_path = output_dir / f"cell_features_{marker}.csv"
        
        if not model_path.exists():
            logger.warning(f"Model file not found for {marker}, skipping...")
            continue
            
        if not feature_path.exists():
            logger.warning(f"Feature file not found for {marker}, skipping...")
            continue
        
        logger.info(f"\U0001f9ec Processing ROI extraction for {marker}")
        process_marker(marker, model_path, feature_path, output_dir)

if __name__ == "__main__":
    main()
