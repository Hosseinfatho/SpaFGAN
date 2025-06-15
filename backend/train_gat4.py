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

# Define markers and their biological interactions
MARKERS = ["CD31", "CD20", "CD11b", "CD4", "CD11c", "Catalase"]
INTERACTIONS = {
    "T-cell entry site": {"CD31": "high", "CD4": "high"},
    "Inflammatory zone": {"CD11b": "high", "CD20": "high"},
    "Oxidative stress niche": {"CD11b": "high", "Catalase": "high"},
    "B-cell infiltration": {"CD20": "high", "CD31": "high"},
    "Dendritic signal": {"CD11c": "high"}
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
            for marker, level in markers.items():
                if marker in MARKERS:
                    marker_idx = MARKERS.index(marker)
                    if marker_idx < x.size(1):  # Ensure index is within bounds
                        if level == "high":
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

def load_graphs(marker):
    """Load graphs for a specific marker and ensure all markers are included"""
    file_path = f"output/roi_graphs_{marker}.pt"
    if not Path(file_path).exists():
        logger.error(f"Graph file not found: {file_path}")
        return None
    
    data = torch.load(file_path)
    graphs = []
    
    # Convert dictionary graphs to PyTorch Geometric Data objects
    for graph_dict in data['graphs']:
        graph = graph_dict['graph']
        if isinstance(graph, dict):
            # Create full feature tensor with all markers
            x = torch.zeros((graph['x'].size(0), len(MARKERS)), device=graph['x'].device)
            
            # Get the marker index
            marker_idx = MARKERS.index(marker)
            
            # Copy the original features to the correct position
            x[:, marker_idx] = graph['x'].squeeze()
            
            # Convert dictionary to Data object with full features
            graph = Data(
                x=x,
                edge_index=graph['edge_index'],
                edge_attr=graph['edge_attr'],
                pos=graph['pos'],
                marker=marker
            )
        graphs.append(graph)
    
    return graphs

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
    
    for batch in tqdm(loader, desc="Training"):
        batch = batch.to(device)
        optimizer.zero_grad()
        
        # Forward pass
        out = model(batch.x, batch.edge_index, batch.edge_attr)
        
        # Calculate loss (reconstruction loss)
        loss = nn.MSELoss()(out, batch.x)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(loader)

def evaluate_model(model, loader, device):
    """Evaluate model on test set"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index, batch.edge_attr)
            
            # Calculate loss
            loss = nn.MSELoss()(out, batch.x)
            total_loss += loss.item()
            
            # Convert predictions to binary (high/low expression)
            preds = (out > batch.x.mean(dim=0)).float()
            labels = (batch.x > batch.x.mean(dim=0)).float()
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate metrics
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    accuracy = accuracy_score(all_labels.flatten(), all_preds.flatten())
    precision = precision_score(all_labels.flatten(), all_preds.flatten(), average='weighted')
    recall = recall_score(all_labels.flatten(), all_preds.flatten(), average='weighted')
    f1 = f1_score(all_labels.flatten(), all_preds.flatten(), average='weighted')
    
    return {
        'loss': total_loss / len(loader),
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

def train_model(marker, num_epochs=100, batch_size=32, learning_rate=0.001):
    """Train GAT model for a specific marker"""
    logger.info(f"\nTraining GAT model for {marker}")
    
    # Load graphs
    graphs = load_graphs(marker)
    if not graphs:
        return
    
    logger.info(f"Loaded {len(graphs)} graphs for {marker}")
    
    # Prepare data loaders
    train_loader, test_loader = prepare_dataloaders(graphs, batch_size)
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
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, device)
        
        # Evaluate
        test_metrics = evaluate_model(model, test_loader, device)
        
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
    
    logger.info(f"\nTraining completed for {marker}")
    logger.info(f"Best Test Loss: {best_loss:.4f}")
    logger.info(f"Best Test Accuracy: {best_metrics['accuracy']:.4f}")
    logger.info(f"Best Test F1 Score: {best_metrics['f1']:.4f}")

def main():
    """Main training function"""
    for marker in MARKERS:
        train_model(marker)
    
    logger.info("\nTraining completed for all markers")

if __name__ == "__main__":
    main() 