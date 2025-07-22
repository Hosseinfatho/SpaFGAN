import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATConv
from torch_geometric.data import Data
import logging
from pathlib import Path
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define markers and their biological interactions - updated
MARKERS = ["CD31", "CD20", "CD11b", "CD4", "CD11c", "Catalase"]
INTERACTIONS = {
    "B-cell infiltration": ["CD20", "CD31"],      # CD20⁺ CD31⁺
    "T-cell maturation": ["CD4", "CD31"],         # CD4⁺ CD31⁺
    "Inflammatory zone": ["CD11b", "CD11c"],      # CD11b⁺ CD11c⁺
    "Oxidative stress regulation": ["CD11b", "Catalase"]  # CD11b⁺ Catalase⁺
}

class MarkerGAT(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=4):
        super(MarkerGAT, self).__init__()
        self.gat1 = GATConv(in_channels, hidden_channels, heads=heads)
        self.gat2 = GATConv(hidden_channels * heads, out_channels, heads=1)
        self.final = nn.Linear(out_channels, in_channels)
        
        # Interaction attention layers
        self.interaction_attention = nn.ModuleDict()
        for interaction_name, markers in INTERACTIONS.items():
            self.interaction_attention[interaction_name] = nn.MultiheadAttention(
                embed_dim=hidden_channels * heads,  # Match GAT output dimension
                num_heads=1,
                batch_first=True
            )
        
    def forward(self, x, edge_index, edge_attr):
        # First GAT layer with edge attributes
        x = torch.relu(self.gat1(x, edge_index, edge_attr))
        
        # Apply interaction attention
        interaction_outputs = []
        for interaction_name, markers in INTERACTIONS.items():
            # Create attention mask based on interaction markers
            mask = torch.zeros(x.size(0), x.size(0), device=x.device)
            for marker in markers:
                if marker in MARKERS:
                    marker_idx = MARKERS.index(marker)
                    if marker_idx < x.size(1):  # Ensure index is within bounds
                        # Set attention mask for nodes with high marker expression
                        node_mask = (x[:, marker_idx] > x[:, marker_idx].mean()).float()
                        mask = mask + torch.outer(node_mask, node_mask)
            
            # Normalize mask
            mask = mask / (mask.sum(dim=1, keepdim=True) + 1e-8)
            
            # Apply attention
            attn_output, _ = self.interaction_attention[interaction_name](
                x, x, x,
                attn_mask=mask
            )
            interaction_outputs.append(attn_output)
        
        # Combine interaction outputs
        if interaction_outputs:
            x = x + torch.stack(interaction_outputs).mean(dim=0)
        
        # Second GAT layer
        x = torch.relu(self.gat2(x, edge_index))
        
        # Final layer to match input dimensions
        x = self.final(x)
        return x

def load_main_graph(marker):
    """Load main graph for a specific marker"""
    file_path = f"output/main_marker_graphs/{marker}_main_graph.pt"
    if not Path(file_path).exists():
        logger.error(f"Main graph file not found: {file_path}")
        return None
    
    main_graph = torch.load(file_path)
    logger.info(f"Loaded main graph for {marker}: {main_graph.metadata['num_nodes']} nodes, {main_graph.metadata['num_edges']} edges")
    return main_graph

def extract_subgraphs_from_main(main_graph, max_subgraphs_per_batch=10):
    """Extract subgraphs from main graph for training"""
    subgraphs = []
    metadata = main_graph.metadata
    
    for subgraph_info in metadata['subgraphs']:
        start_node = subgraph_info['start_node']
        end_node = subgraph_info['end_node']
        num_total_nodes = subgraph_info['num_total_nodes']  # Use num_total_nodes instead of num_nodes
        
        # Extract node features for this subgraph
        subgraph_x = main_graph.x[start_node:end_node + 1]
        
        # Extract edges that are within this subgraph
        edge_mask = ((main_graph.edge_index[0] >= start_node) & 
                    (main_graph.edge_index[0] <= end_node) &
                    (main_graph.edge_index[1] >= start_node) & 
                    (main_graph.edge_index[1] <= end_node))
        
        subgraph_edge_index = main_graph.edge_index[:, edge_mask]
        subgraph_edge_attr = main_graph.edge_attr[edge_mask]
        
        # Adjust edge indices to be local to this subgraph
        subgraph_edge_index = subgraph_edge_index - start_node
        
        # Create subgraph Data object
        subgraph = Data(
            x=subgraph_x,
            edge_index=subgraph_edge_index,
            edge_attr=subgraph_edge_attr,
            roi_id=subgraph_info['roi_id'],
            center=subgraph_info['center']
        )
        
        subgraphs.append(subgraph)
    
    logger.info(f"Extracted {len(subgraphs)} subgraphs from main graph")
    return subgraphs

def prepare_dataloaders(graphs, batch_size=32, test_size=0.15):
    """Prepare train and test DataLoaders"""
    # Shuffle graphs
    np.random.shuffle(graphs)
    
    # Split into train and test
    test_size = int(len(graphs) * test_size)
    train_graphs = graphs[test_size:]
    test_graphs = graphs[:test_size]
    
    # Create DataLoaders
    train_loader = DataLoader(train_graphs, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_graphs, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader

def train_epoch(model, loader, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    valid_batches = 0
    
    for batch in tqdm(loader, desc="Training"):
        try:
            batch = batch.to(device)
            optimizer.zero_grad()
            
            # Forward pass
            out = model(batch.x, batch.edge_index, batch.edge_attr)
            
            # Calculate loss (reconstruction loss)
            loss = nn.MSELoss()(out, batch.x)
            
            # Check for invalid loss
            if np.isnan(loss.item()) or np.isinf(loss.item()):
                logger.warning(f"Invalid loss in training: {loss.item()}")
                continue
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            total_loss += loss.item()
            valid_batches += 1
            
        except Exception as e:
            logger.warning(f"Error in training batch: {str(e)}")
            continue
    
    return total_loss / valid_batches if valid_batches > 0 else float('inf')

def evaluate_model(model, loader, device):
    """Evaluate model on test set"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    try:
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(device)
                out = model(batch.x, batch.edge_index, batch.edge_attr)
                
                # Calculate loss
                loss = nn.MSELoss()(out, batch.x)
                
                # Check for invalid loss
                if np.isnan(loss.item()) or np.isinf(loss.item()):
                    logger.warning(f"Invalid loss in evaluation: {loss.item()}")
                    continue
                
                total_loss += loss.item()
                
                # Convert predictions to binary (high/low expression)
                preds = (out > batch.x.mean(dim=0)).float()
                labels = (batch.x > batch.x.mean(dim=0)).float()
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Check if we have any valid predictions
        if len(all_preds) == 0:
            logger.warning("No valid predictions in evaluation")
            return {
                'loss': float('inf'),
                'accuracy': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'f1': 0.0
            }
        
        # Calculate metrics
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        
        # Check for NaN or Inf values
        if np.isnan(all_preds).any() or np.isinf(all_preds).any():
            logger.warning("NaN or Inf values in predictions")
            return {
                'loss': float('inf'),
                'accuracy': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'f1': 0.0
            }
        
        accuracy = accuracy_score(all_labels.flatten(), all_preds.flatten())
        precision = precision_score(all_labels.flatten(), all_preds.flatten(), average='weighted', zero_division=0)
        recall = recall_score(all_labels.flatten(), all_preds.flatten(), average='weighted', zero_division=0)
        f1 = f1_score(all_labels.flatten(), all_preds.flatten(), average='weighted', zero_division=0)
        
        return {
            'loss': total_loss / len(loader) if len(loader) > 0 else float('inf'),
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
        
    except Exception as e:
        logger.error(f"Error in evaluation: {str(e)}")
        return {
            'loss': float('inf'),
            'accuracy': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0
        }

def train_model(marker, num_epochs=100, batch_size=32, learning_rate=0.001):
    """Train GAT model for a specific marker"""
    logger.info(f"\nTraining GAT model for {marker}")
    
    # Load main graph
    main_graph = load_main_graph(marker)
    if not main_graph:
        return
    
    # Special handling for CD11c (nodes only, no edges)
    if marker == "CD11c":
        logger.info(f"Training CD11c with nodes only (no edges)")
        return train_cd11c_nodes_only(main_graph, num_epochs, batch_size, learning_rate)
    
    # Extract subgraphs
    subgraphs = extract_subgraphs_from_main(main_graph)
    
    # Check if we have enough subgraphs
    if len(subgraphs) < 2:
        logger.warning(f"Not enough subgraphs for {marker}: {len(subgraphs)}")
        return
    
    # Prepare data loaders
    train_loader, test_loader = prepare_dataloaders(subgraphs, batch_size)
    logger.info(f"Train set size: {len(train_loader.dataset)}, Test set size: {len(test_loader.dataset)}")
    
    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    model = MarkerGAT(
        in_channels=len(MARKERS),
        hidden_channels=64,
        out_channels=32
    ).to(device)
    
    # Initialize optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    best_loss = float('inf')
    best_metrics = None
    
    for epoch in range(num_epochs):
        try:
            # Train
            train_loss = train_epoch(model, train_loader, optimizer, device)
            
            # Check for invalid loss
            if np.isnan(train_loss) or np.isinf(train_loss):
                logger.warning(f"Invalid train loss at epoch {epoch+1}: {train_loss}")
                continue
            
            # Evaluate
            test_metrics = evaluate_model(model, test_loader, device)
            
            # Check for invalid test loss
            if np.isnan(test_metrics['loss']) or np.isinf(test_metrics['loss']):
                logger.warning(f"Invalid test loss at epoch {epoch+1}: {test_metrics['loss']}")
                continue
            
            logger.info(f"Epoch {epoch+1}/{num_epochs}")
            logger.info(f"Train Loss: {train_loss:.4f}")
            logger.info(f"Test Loss: {test_metrics['loss']:.4f}")
            logger.info(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
            logger.info(f"Test F1 Score: {test_metrics['f1']:.4f}")
            
            # Save best model
            if test_metrics['loss'] < best_loss:
                best_loss = test_metrics['loss']
                best_metrics = test_metrics
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': best_loss,
                    'metrics': best_metrics
                }, f"output/best_model_{marker}.pt")
                logger.info(f"Saved new best model for {marker} with loss: {best_loss:.4f}")
                
        except Exception as e:
            logger.error(f"Error in epoch {epoch+1}: {str(e)}")
            continue
    
    logger.info(f"\nTraining completed for {marker}")
    
    if best_metrics is not None:
        logger.info(f"Best Test Loss: {best_loss:.4f}")
        logger.info(f"Best Test Accuracy: {best_metrics['accuracy']:.4f}")
        logger.info(f"Best Test F1 Score: {best_metrics['f1']:.4f}")
    else:
        logger.warning(f"No valid model found for {marker}")
        # Save the last model anyway
        torch.save({
            'epoch': num_epochs - 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': float('inf'),
            'metrics': None
        }, f"output/best_model_{marker}.pt")
        logger.info(f"Saved last model for {marker} (no valid metrics)")

def train_cd11c_nodes_only(main_graph, num_epochs=100, batch_size=32, learning_rate=0.001):
    """Train CD11c model using only nodes and intensities (no edges)"""
    logger.info("Training CD11c with nodes only approach")
    
    # Extract node features from main graph
    node_features = main_graph.x  # [num_nodes, 6] - all marker intensities
    
    # Create simple dataset (no edges needed)
    dataset = []
    for i in range(0, len(node_features), batch_size):
        batch_features = node_features[i:i+batch_size]
        dataset.append(batch_features)
    
    # Split into train and test
    np.random.shuffle(dataset)
    test_size = int(len(dataset) * 0.15)
    train_dataset = dataset[test_size:]
    test_dataset = dataset[:test_size]
    
    logger.info(f"Train set size: {len(train_dataset)}, Test set size: {len(test_dataset)}")
    
    # Initialize simple model for CD11c (no GAT, just MLP)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    class CD11cMLP(nn.Module):
        def __init__(self, in_channels):
            super(CD11cMLP, self).__init__()
            self.fc1 = nn.Linear(in_channels, 64)
            self.fc2 = nn.Linear(64, 32)
            self.fc3 = nn.Linear(32, in_channels)
            self.dropout = nn.Dropout(0.2)
            
        def forward(self, x):
            x = torch.relu(self.fc1(x))
            x = self.dropout(x)
            x = torch.relu(self.fc2(x))
            x = self.dropout(x)
            x = self.fc3(x)
            return x
    
    model = CD11cMLP(in_channels=len(MARKERS)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    best_loss = float('inf')
    best_metrics = None
    
    for epoch in range(num_epochs):
        try:
            # Train
            model.train()
            total_loss = 0
            valid_batches = 0
            
            for batch_features in train_dataset:
                batch_features = batch_features.to(device)
                optimizer.zero_grad()
                
                # Forward pass
                out = model(batch_features)
                
                # Calculate loss (reconstruction loss)
                loss = nn.MSELoss()(out, batch_features)
                
                # Check for invalid loss
                if np.isnan(loss.item()) or np.isinf(loss.item()):
                    continue
                
                # Backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                total_loss += loss.item()
                valid_batches += 1
            
            train_loss = total_loss / valid_batches if valid_batches > 0 else float('inf')
            
            # Evaluate
            model.eval()
            total_loss = 0
            all_preds = []
            all_labels = []
            
            with torch.no_grad():
                for batch_features in test_dataset:
                    batch_features = batch_features.to(device)
                    out = model(batch_features)
                    
                    # Calculate loss
                    loss = nn.MSELoss()(out, batch_features)
                    
                    if np.isnan(loss.item()) or np.isinf(loss.item()):
                        continue
                    
                    total_loss += loss.item()
                    
                    # Convert predictions to binary
                    preds = (out > batch_features.mean(dim=0)).float()
                    labels = (batch_features > batch_features.mean(dim=0)).float()
                    
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
            
            if len(all_preds) > 0:
                all_preds = np.array(all_preds)
                all_labels = np.array(all_labels)
                
                accuracy = accuracy_score(all_labels.flatten(), all_preds.flatten())
                precision = precision_score(all_labels.flatten(), all_preds.flatten(), average='weighted', zero_division=0)
                recall = recall_score(all_labels.flatten(), all_preds.flatten(), average='weighted', zero_division=0)
                f1 = f1_score(all_labels.flatten(), all_preds.flatten(), average='weighted', zero_division=0)
                
                test_metrics = {
                    'loss': total_loss / len(test_dataset) if len(test_dataset) > 0 else float('inf'),
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1': f1
                }
            else:
                test_metrics = {
                    'loss': float('inf'),
                    'accuracy': 0.0,
                    'precision': 0.0,
                    'recall': 0.0,
                    'f1': 0.0
                }
            
            logger.info(f"Epoch {epoch+1}/{num_epochs}")
            logger.info(f"Train Loss: {train_loss:.4f}")
            logger.info(f"Test Loss: {test_metrics['loss']:.4f}")
            logger.info(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
            logger.info(f"Test F1 Score: {test_metrics['f1']:.4f}")
            
            # Save best model
            if test_metrics['loss'] < best_loss:
                best_loss = test_metrics['loss']
                best_metrics = test_metrics
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': best_loss,
                    'metrics': best_metrics,
                    'model_type': 'CD11cMLP'
                }, f"output/best_model_CD11c.pt")
                logger.info(f"Saved new best model for CD11c with loss: {best_loss:.4f}")
                
        except Exception as e:
            logger.error(f"Error in epoch {epoch+1}: {str(e)}")
            continue
    
    logger.info(f"\nTraining completed for CD11c")
    
    if best_metrics is not None:
        logger.info(f"Best Test Loss: {best_loss:.4f}")
        logger.info(f"Best Test Accuracy: {best_metrics['accuracy']:.4f}")
        logger.info(f"Best Test F1 Score: {best_metrics['f1']:.4f}")
    else:
        logger.warning(f"No valid model found for CD11c")
        # Save the last model anyway
        torch.save({
            'epoch': num_epochs - 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': float('inf'),
            'metrics': None,
            'model_type': 'CD11cMLP'
        }, f"output/best_model_CD11c.pt")
        logger.info(f"Saved last model for CD11c (no valid metrics)")

def main():
    """Main training function"""
    for marker in MARKERS:
        train_model(marker)
    
    logger.info("\nTraining completed for all markers")

if __name__ == "__main__":
    main() 