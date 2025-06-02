import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
import logging

logger = logging.getLogger(__name__)

class SpaFGAN(nn.Module):
    def __init__(self, in_channels, hidden_channels=32, out_channels=1, heads=2, dropout=0.2):
        """
        Graph Attention Network for spatial feature analysis
        
        Args:
            in_channels (int): Number of input features
            hidden_channels (int): Number of hidden units
            out_channels (int): Number of output classes
            heads (int): Number of attention heads
            dropout (float): Dropout rate
        """
        super(SpaFGAN, self).__init__()
        
        # First GAT layer with multiple heads
        self.gat1 = GATConv(
            in_channels=in_channels,
            out_channels=hidden_channels,
            heads=heads,
            dropout=dropout,
            concat=True
        )
        
        # Second GAT layer with single head
        self.gat2 = GATConv(
            in_channels=hidden_channels * heads,
            out_channels=hidden_channels,
            heads=1,
            dropout=dropout,
            concat=False
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(hidden_channels * heads)
        self.norm2 = nn.LayerNorm(hidden_channels)
        
        # Output layer
        self.out = nn.Linear(hidden_channels, out_channels)
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout)
        
        logger.info(f"Initialized SpaFGAN model:")
        logger.info(f"Input dimension: {in_channels}")
        logger.info(f"Hidden dimension: {hidden_channels}")
        logger.info(f"Number of attention heads: {heads}")
    
    def forward(self, x, edge_index, edge_weights=None):
        """
        Forward pass
        
        Args:
            x (torch.Tensor): Node features [N, in_channels]
            edge_index (torch.Tensor): Graph connectivity [2, E]
            edge_weights (torch.Tensor, optional): Edge weights [E]
            
        Returns:
            torch.Tensor: Node predictions [N, out_channels]
        """
        # First GAT layer with multi-head attention
        x = self.gat1(x, edge_index, edge_weights)
        x = self.norm1(x)
        x = F.elu(x)
        x = self.dropout(x)
        
        # Second GAT layer
        x = self.gat2(x, edge_index, edge_weights)
        x = self.norm2(x)
        x = F.elu(x)
        x = self.dropout(x)
        
        # Output layer
        x = self.out(x)
        
        return x
    
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
            return self(data.x, data.edge_index, data.edge_weights if hasattr(data, 'edge_weights') else None) 