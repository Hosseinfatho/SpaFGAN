import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import from_networkx
import logging
from models.gat import SpaFGAN
from torch.optim import Adam
import numpy as np

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_and_prepare_graph(graph_path="backend/output/spatial_graph.pkl"):
    """
    Load the spatial graph and convert it to PyTorch Geometric format.
    
    Args:
        graph_path (str): Path to the pickle file containing the spatial graph
        
    Returns:
        torch_geometric.data.Data: Graph in PyG format
    """
    try:
        # Load spatial graph
        logger.info(f"Loading spatial graph from {graph_path}")
        with open(graph_path, "rb") as f:
            G = pickle.load(f)
        
        # Prepare node features
        logger.info("Preparing node features")
        for n in G.nodes:
            G.nodes[n]["x"] = torch.tensor(G.nodes[n]["features"], dtype=torch.float32)
        
        # Convert to PyG format
        logger.info("Converting to PyTorch Geometric format")
        data = from_networkx(G, group_node_attrs=["x"])
        
        # Move to GPU if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")
        data = data.to(device)
        
        # Add temporary dummy labels
        N = data.num_nodes
        data.y = torch.randint(0, 2, (N,), dtype=torch.float32).to(device)
        logger.info(f"Added temporary binary labels for {N} nodes")
        
        # Log graph information
        logger.info(f"Graph loaded successfully:")
        logger.info(f"Number of nodes: {data.num_nodes}")
        logger.info(f"Number of edges: {data.num_edges}")
        logger.info(f"Node feature dimension: {data.x.size(1)}")
        
        return data
        
    except FileNotFoundError:
        logger.error(f"Graph file not found at {graph_path}")
        raise
    except Exception as e:
        logger.error(f"Error loading graph: {str(e)}")
        raise

def train_model(model, data, optimizer, loss_fn, epochs=100):
    """
    Train the SpaFGAN model.
    
    Args:
        model (SpaFGAN): The model to train
        data (torch_geometric.data.Data): Training data
        optimizer (torch.optim.Optimizer): Optimizer
        loss_fn (nn.Module): Loss function
        epochs (int): Number of training epochs
    """
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # Forward pass
        out = model(data.x, data.edge_index)
        
        # Compute loss
        loss = loss_fn(out, data.y)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            logger.info(f'Epoch {epoch+1}/{epochs} | Loss: {loss.item():.4f}')

def main():
    try:
        # Load and prepare data
        data = load_and_prepare_graph()
        
        # Initialize model
        model = SpaFGAN().to(data.x.device)
        
        # Initialize optimizer and loss function
        optimizer = Adam(model.parameters(), lr=0.01)
        loss_fn = nn.BCELoss()
        
        # Train model
        logger.info("Starting model training...")
        train_model(model, data, optimizer, loss_fn)
        
        # Save model
        torch.save(model.state_dict(), "backend/output/spafgan_model.pt")
        logger.info("Model saved successfully")
        
        # Save predictions
        model.eval()
        with torch.no_grad():
            predictions = model(data.x, data.edge_index)
            torch.save(predictions, "backend/output/roi_scores.pt")
        logger.info("ROI scores saved successfully")
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise

if __name__ == "__main__":
    main() 