import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from models.gat import SpaFGAN
from torch.optim import Adam, AdamW
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
MARKERS = ["CD31", "CD20", "CD11b", "CD4", "CD11c", "Catalase"]

# Define target markers for training
TARGET_MARKERS = ["CD31", "CD11b", "CD11c"]

def create_spatial_graph(features, positions, radius=30):
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

def prepare_data(feature_path, marker):
    """
    Prepare data with feature selection and better handling of CD11b
    """
    # Load cell features
    df = pd.read_csv(feature_path)
    logger.info(f"Loaded {len(df)} cells for {marker}")
    
    # Validate data
    if len(df) == 0:
        raise ValueError(f"No cells found in {feature_path}")
    
    # Check for required columns
    required_columns = MARKERS + ['x', 'y', 'z']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing columns in {feature_path}: {missing_columns}")
    
    # Prepare features (all marker intensities)
    features = df[MARKERS].values
    
    # Normalize marker intensities
    marker_scaler = MinMaxScaler()
    features = marker_scaler.fit_transform(features)
    
    # Add spatial features with more complex interactions
    positions = df[['x', 'y', 'z']].values
    spatial_features = np.concatenate([
        positions,
        positions ** 2,  # Quadratic terms
        positions ** 3,  # Cubic terms
        np.prod(positions, axis=1, keepdims=True),  # Interaction terms
        np.sin(positions),  # Periodic features
        np.cos(positions)   # Periodic features
    ], axis=1)
    
    # Normalize spatial features
    spatial_scaler = MinMaxScaler()
    spatial_features = spatial_scaler.fit_transform(spatial_features)
    
    # Add relative position features
    center = positions.mean(axis=0)
    relative_pos = positions - center
    relative_features = np.concatenate([
        relative_pos,
        np.linalg.norm(relative_pos, axis=1, keepdims=True),  # Distance from center
        relative_pos / (np.linalg.norm(relative_pos, axis=1, keepdims=True) + 1e-6)  # Normalized direction
    ], axis=1)
    
    # Normalize relative features
    relative_scaler = MinMaxScaler()
    relative_features = relative_scaler.fit_transform(relative_features)
    
    # Combine all features
    all_features = np.concatenate([
        features,
        spatial_features,
        relative_features
    ], axis=1)
    
    # Final normalization
    scaler = MinMaxScaler()
    all_features = scaler.fit_transform(all_features)
    
    # Feature selection for CD11b
    if marker == "CD11b":
        # Select features with high variance
        feature_variances = np.var(all_features, axis=0)
        selected_features = feature_variances > np.percentile(feature_variances, 50)
        all_features = all_features[:, selected_features]
        logger.info(f"Selected {selected_features.sum()} features for CD11b")
    
    all_features = torch.tensor(all_features, dtype=torch.float32)
    
    # Get spatial positions for graph construction
    positions = torch.tensor(positions, dtype=torch.float32)
    
    # Create spatial graph with weighted edges
    edge_index, edge_weights = create_spatial_graph(all_features, positions, radius=30)
    logger.info(f"Created graph with {edge_index.shape[1]} edges for {marker}")
    
    # Create labels based on the primary marker
    primary_marker = df[marker].values
    mean = primary_marker.mean()
    std = primary_marker.std()
    normalized = (primary_marker - mean) / (std + 1e-6)
    
    # Use adaptive threshold for CD11b
    if marker == "CD11b":
        # Use multiple thresholds and take the best
        thresholds = [0.1, 0.2, 0.3, 0.4, 0.5]
        best_threshold = 0.1
        best_balance = float('inf')
        
        for threshold in thresholds:
            labels_temp = (normalized > threshold).astype(float)
            pos_count = labels_temp.sum()
            neg_count = len(labels_temp) - pos_count
            balance = abs(pos_count - neg_count)
            
            if balance < best_balance:
                best_balance = balance
                best_threshold = threshold
        
        threshold = best_threshold
        logger.info(f"Selected threshold {threshold:.2f} for CD11b")
    else:
        threshold = 0.1  # Default threshold
    
    labels = (normalized > threshold).astype(float)
    
    # Apply label smoothing
    smoothing = 0.1
    labels = labels * (1 - smoothing) + smoothing / 2
    labels = torch.tensor(labels, dtype=torch.float32)
    
    # Log label distribution
    positive_count = (labels > 0.5).sum().item()
    negative_count = len(labels) - positive_count
    logger.info(f"Label distribution for {marker}:")
    logger.info(f"Positive cells: {positive_count} ({positive_count/len(labels)*100:.1f}%)")
    logger.info(f"Negative cells: {negative_count} ({negative_count/len(labels)*100:.1f}%)")
    
    # Calculate class weights for balanced training
    pos_weight = negative_count / (positive_count + 1e-6)
    neg_weight = positive_count / (negative_count + 1e-6)
    class_weights = torch.tensor([neg_weight, pos_weight], dtype=torch.float32)
    logger.info(f"Class weights: negative={neg_weight:.2f}, positive={pos_weight:.2f}")
    
    # Create train/val split with stratification
    train_idx, val_idx = train_test_split(
        np.arange(len(labels)),
        test_size=0.2,
        random_state=42,
        stratify=(labels > 0.5).numpy()
    )
    
    # Create masks
    train_mask = torch.zeros(len(labels), dtype=torch.bool)
    val_mask = torch.zeros(len(labels), dtype=torch.bool)
    train_mask[train_idx] = True
    val_mask[val_idx] = True
    
    # Log split distribution
    train_positive = (labels[train_mask] > 0.5).sum().item()
    val_positive = (labels[val_mask] > 0.5).sum().item()
    logger.info(f"Train set: {len(train_idx)} cells ({train_positive} positive)")
    logger.info(f"Val set: {len(val_idx)} cells ({val_positive} positive)")
    
    return all_features, edge_index, edge_weights, labels, train_mask, val_mask, class_weights

def focal_loss(pred, target, alpha=0.25, gamma=2.0):
    """
    Focal Loss for better handling of class imbalance
    
    Args:
        pred (torch.Tensor): Model predictions
        target (torch.Tensor): True labels
        alpha (float): Weighting factor for positive class
        gamma (float): Focusing parameter
        
    Returns:
        torch.Tensor: Focal loss value
    """
    # Clip predictions for numerical stability
    pred = torch.clamp(pred, min=-10, max=10)
    
    # Calculate BCE loss
    bce_loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
    
    # Calculate focal loss components
    pt = torch.exp(-bce_loss)
    focal_weight = alpha * (1-pt)**gamma
    
    # Apply focal weight and take mean
    focal_loss = (focal_weight * bce_loss).mean()
    
    return focal_loss

def train_model(model, features, edge_index, edge_weights, labels, train_mask, val_mask, marker, class_weights, num_epochs=300):
    """
    Train the model with improved loss calculation
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    model = model.to(device)
    features = features.to(device)
    edge_index = edge_index.to(device)
    edge_weights = edge_weights.to(device)
    labels = labels.to(device).unsqueeze(1)
    train_mask = train_mask.to(device)
    val_mask = val_mask.to(device)
    class_weights = class_weights.to(device)
    
    # Use AdamW optimizer with lower learning rate
    optimizer = AdamW(model.parameters(), lr=0.0005, weight_decay=0.005)
    
    # Use cosine annealing with warm restarts
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=50,
        T_mult=2,
        eta_min=1e-6
    )
    
    best_val_loss = float('inf')
    patience = 100
    patience_counter = 0
    
    # Initialize best model state and metrics
    best_model_state = None
    best_metrics = {
        'train_loss': float('inf'),
        'val_loss': float('inf'),
        'train_acc': 0.0,
        'val_acc': 0.0,
        'epoch': 0
    }
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': []
    }
    
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        
        # Forward pass with edge weights
        out = model(features, edge_index, edge_weights)
        
        # Calculate weighted BCE loss with label smoothing
        train_loss = F.binary_cross_entropy_with_logits(
            out[train_mask],
            labels[train_mask],
            pos_weight=class_weights[1],
            reduction='mean'
        )
        
        # Add moderate L2 regularization
        l2_lambda = 0.005
        l2_reg = torch.tensor(0., device=device)
        for param in model.parameters():
            l2_reg += torch.norm(param)
        train_loss = train_loss + l2_lambda * l2_reg
        
        # Backward pass
        train_loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
        
        optimizer.step()
        scheduler.step()
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_out = model(features, edge_index, edge_weights)
            val_loss = F.binary_cross_entropy_with_logits(
                val_out[val_mask],
                labels[val_mask],
                pos_weight=class_weights[1],
                reduction='mean'
            )
        
        # Calculate metrics
        train_pred = (out[train_mask] > 0).float()
        train_acc = (train_pred == (labels[train_mask] > 0.5).float()).float().mean()
        val_pred = (val_out[val_mask] > 0).float()
        val_acc = (val_pred == (labels[val_mask] > 0.5).float()).float().mean()
        
        # Store metrics
        history['train_loss'].append(train_loss.item())
        history['val_loss'].append(val_loss.item())
        history['train_acc'].append(train_acc.item())
        history['val_acc'].append(val_acc.item())
        
        # Logging
        logger.info(
            f'[{marker}] Epoch {epoch:03d}, '
            f'Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, '
            f'Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}, '
            f'LR: {scheduler.get_last_lr()[0]:.6f}'
        )
        
        # Early stopping with best model saving
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
            
            # Update best metrics
            best_metrics = {
                'train_loss': train_loss.item(),
                'val_loss': val_loss.item(),
                'train_acc': train_acc.item(),
                'val_acc': val_acc.item(),
                'epoch': epoch
            }
            
            # Save best model
            model_path = Path(__file__).parent / "models" / f"spafgan_{marker}.pth"
            model_path.parent.mkdir(exist_ok=True)
            torch.save(best_model_state, model_path)
            logger.info(f"Saved best model for {marker} at epoch {epoch}")
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            logger.info(f"Early stopping for {marker} at epoch {epoch}")
            break
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    # Print final results for paper
    logger.info(f"\n{'='*50}")
    logger.info(f"Final Results for {marker}:")
    logger.info(f"{'='*50}")
    logger.info(f"Best Epoch: {best_metrics['epoch']}")
    logger.info(f"Training Loss: {best_metrics['train_loss']:.4f}")
    logger.info(f"Validation Loss: {best_metrics['val_loss']:.4f}")
    logger.info(f"Training Accuracy: {best_metrics['train_acc']:.4f}")
    logger.info(f"Validation Accuracy: {best_metrics['val_acc']:.4f}")
    logger.info(f"Number of Cells: {len(labels)}")
    logger.info(f"Number of Edges: {edge_index.shape[1]}")
    logger.info(f"{'='*50}\n")
    
    return model, best_metrics

def main():
    # Get the absolute path to the backend directory
    backend_dir = Path(__file__).parent.resolve()
    input_dir = backend_dir / "input"
    output_dir = backend_dir / "output"
    
    # Store results for all markers
    all_results = {}
    
    # Train model for each marker
    for marker in TARGET_MARKERS:
        logger.info(f"\n{'='*50}")
        logger.info(f"Training model for {marker}")
        logger.info(f"{'='*50}")
        
        # Prepare data
        feature_path = output_dir / f"cell_features_{marker}.csv"
        if not feature_path.exists():
            logger.error(f"Feature file not found: {feature_path}")
            continue
            
        features, edge_index, edge_weights, labels, train_mask, val_mask, class_weights = prepare_data(
            feature_path, marker
        )
        
        # Create model
        model = SpaFGAN(
            in_channels=features.shape[1],
            hidden_channels=64,
            out_channels=1,
            heads=2,
            dropout=0.2
        )
        
        # Train model
        model, metrics = train_model(
            model, features, edge_index, edge_weights,
            labels, train_mask, val_mask, marker, class_weights
        )
        
        # Store results with correct cell count
        all_results[marker] = {
            **metrics,
            'num_cells': len(labels),
            'num_edges': edge_index.shape[1]
        }
        
        logger.info(f"Completed training for {marker}\n")
    
    # Print summary table for paper
    logger.info("\nSummary of Results for Paper:")
    logger.info("="*80)
    logger.info(f"{'Marker':<10} {'Cells':<10} {'Train Acc':<12} {'Val Acc':<12} {'Train Loss':<12} {'Val Loss':<12}")
    logger.info("-"*80)
    for marker, metrics in all_results.items():
        logger.info(
            f"{marker:<10} {metrics['num_cells']:<10} "
            f"{metrics['train_acc']:.4f}      {metrics['val_acc']:.4f}      "
            f"{metrics['train_loss']:.4f}      {metrics['val_loss']:.4f}"
        )
    logger.info("="*80)
    
    # Print detailed cell counts
    logger.info("\nDetailed Cell Counts:")
    logger.info("="*50)
    for marker, metrics in all_results.items():
        logger.info(f"{marker}: {metrics['num_cells']} cells, {metrics['num_edges']} edges")
    logger.info("="*50)

if __name__ == "__main__":
    logger.info("Starting SpaFGAN training for all markers...")
    main()
    logger.info("Training completed for all markers.") 