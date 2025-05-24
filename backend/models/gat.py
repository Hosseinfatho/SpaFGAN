import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
import logging

logger = logging.getLogger(__name__)

class SpaFGAN(nn.Module):
    def __init__(self, in_dim=6, hidden=32, heads=2):
        """
        Initialize the SpaFGAN model.
        
        Args:
            in_dim (int): Number of input features
            hidden (int): Number of hidden channels
            heads (int): Number of attention heads
        """
        super(SpaFGAN, self).__init__()
        self.gat1 = GATConv(in_dim, hidden, heads=heads)
        self.gat2 = GATConv(hidden * heads, hidden, heads=1)
        self.out = nn.Linear(hidden, 1)
        
        logger.info(f"Initialized SpaFGAN model:")
        logger.info(f"Input dimension: {in_dim}")
        logger.info(f"Hidden dimension: {hidden}")
        logger.info(f"Number of attention heads: {heads}")
    
    def forward(self, x, edge_index):
        """
        Forward pass of the model.
        
        Args:
            x (torch.Tensor): Node features
            edge_index (torch.Tensor): Graph connectivity
            
        Returns:
            torch.Tensor: ROI scores for each node
        """
        x = F.relu(self.gat1(x, edge_index))
        x = F.relu(self.gat2(x, edge_index))
        return torch.sigmoid(self.out(x)).squeeze()
    
    def predict(self, data):
        """
        Make predictions on a graph.
        
        Args:
            data (torch_geometric.data.Data): Input graph data
            
        Returns:
            torch.Tensor: ROI scores for each node
        """
        self.eval()
        with torch.no_grad():
            return self(data.x, data.edge_index) 