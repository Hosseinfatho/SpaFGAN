import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from models.gat import SpaFGAN
from torch.optim import Adam
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path
from torch_geometric.utils import dense_to_sparse
import networkx as nx
from sklearn.preprocessing import MinMaxScaler

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define markers
MARKERS = ["CD31", "CD20", "CD11b", "CD4", "Catalase"]

def create_spatial_graph(features, positions, radius=50):
    """
    Create a spatial graph based on radius-based neighborhood with weighted edges
    
    Args:
        features (torch.Tensor): Node features
        positions (torch.Tensor): Node positions (x, y, z)
        radius (float): Maximum distance for connecting nodes FFN
        
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

def prepare_data(feature_path, marker):
    """
    Prepare data from cell features CSV file and create spatial graph
    
    Args:
        feature_path (str): Path to the cell features CSV file
        marker (str): Name of the marker
        
    Returns:
        tuple: (features, edge_index, edge_weights, labels, train_mask, val_mask)
    """
    # Load cell features
    df = pd.read_csv(feature_path)
    
    # Prepare features (all marker intensities)
    features = df[MARKERS].values
    
    # Add spatial features
    positions = df[['x', 'y', 'z']].values
    spatial_features = np.concatenate([
        positions,
        positions ** 2,  # Quadratic terms
        np.prod(positions, axis=1, keepdims=True)  # Interaction terms
    ], axis=1)
    
    # Combine marker and spatial features
    all_features = np.concatenate([features, spatial_features], axis=1)
    
    # Normalize features using MinMaxScaler
    scaler = MinMaxScaler()
    all_features = scaler.fit_transform(all_features)
    all_features = torch.tensor(all_features, dtype=torch.float32)
    
    # Get spatial positions for graph construction
    positions = torch.tensor(positions, dtype=torch.float32)
    
    # Create spatial graph with weighted edges
    edge_index, edge_weights = create_spatial_graph(all_features, positions, radius=50)
    
    # Create labels based on the primary marker
    primary_marker = df[marker].values
    mean = primary_marker.mean()
    std = primary_marker.std()
    normalized = (primary_marker - mean) / (std + 1e-6)
    
    # Adjust threshold based on marker
    threshold = 0.3 if marker == "CD11b" else 0.5
    labels = (normalized > threshold).astype(float)
    labels = torch.tensor(labels, dtype=torch.float32)
    
    # Create train/val split
    train_idx, val_idx = train_test_split(
        np.arange(len(labels)),
        test_size=0.2,
        random_state=42,
        stratify=labels.numpy()
    )
    
    # Create masks
    train_mask = torch.zeros(len(labels), dtype=torch.bool)
    val_mask = torch.zeros(len(labels), dtype=torch.bool)
    train_mask[train_idx] = True
    val_mask[val_idx] = True
    
    return all_features, edge_index, edge_weights, labels, train_mask, val_mask

def train_model(model, features, edge_index, edge_weights, labels, train_mask, val_mask, marker, num_epochs=100):
    """
    Train the model with early stopping and validation
    
    Args:
        model: The SpaFGAN model
        features: Input features
        edge_index: Graph connectivity
        edge_weights: Edge weights
        labels: Target labels
        train_mask: Training mask
        val_mask: Validation mask
        marker: Marker name for logging
        num_epochs: Number of epochs to train
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    features = features.to(device)
    edge_index = edge_index.to(device)
    edge_weights = edge_weights.to(device)
    labels = labels.to(device).unsqueeze(1)
    train_mask = train_mask.to(device)
    val_mask = val_mask.to(device)
    
    optimizer = Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
    best_val_loss = float('inf')
    patience = 30
    patience_counter = 0
    
    # Initialize learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.7, patience=10, verbose=True, min_lr=1e-6
    )
    
    # Initialize best model state
    best_model_state = None
    
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        
        # Forward pass with edge weights
        out = model(features, edge_index, edge_weights)
        
        # Calculate loss
        train_loss = F.binary_cross_entropy_with_logits(
            out[train_mask],
            labels[train_mask]
        )
        
        # Backward pass
        train_loss.backward()
        optimizer.step()
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_out = model(features, edge_index, edge_weights)
            val_loss = F.binary_cross_entropy_with_logits(
                val_out[val_mask],
                labels[val_mask]
            )
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Logging
        train_pred = (out[train_mask] > 0).float()
        train_acc = (train_pred == labels[train_mask]).float().mean()
        val_pred = (val_out[val_mask] > 0).float()
        val_acc = (val_pred == labels[val_mask]).float().mean()
        
        logger.info(
            f'Epoch {epoch:03d}, '
            f'Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, '
            f'Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}, '
            f'LR: {optimizer.param_groups[0]["lr"]:.6f}'
        )
        
        # Early stopping with best model saving
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping triggered after {epoch} epochs")
                break
    
    # Save best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        torch.save(best_model_state, f"output/best_spafgan_model_{marker}.pt")
        logger.info(f"Saved best model with validation loss: {best_val_loss:.4f}")

def main():
    try:
        backend_dir = Path(__file__).parent.resolve()
        output_dir = backend_dir / "output"
        
        for marker in MARKERS:
            logger.info(f"Training model for marker: {marker}")
            feature_path = output_dir / f"cell_features_{marker}.csv"
            
            if not feature_path.exists():
                logger.warning(f"Feature file not found for {marker}, skipping...")
                continue
            
            # Prepare data
            features, edge_index, edge_weights, labels, train_mask, val_mask = prepare_data(feature_path, marker)
            
            # Create model
            model = SpaFGAN(
                in_channels=features.size(1),  # Updated input channels
                hidden_channels=32,
                out_channels=1,
                heads=2
            )
            
            # Train model
            train_model(model, features, edge_index, edge_weights, labels, train_mask, val_mask, marker)
            
    except Exception as e:
        logger.error(f"Error during training: {str(e)}")
        raise

if __name__ == "__main__":
    main() 