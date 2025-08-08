import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
import logging

logger = logging.getLogger(__name__)

class SSGAT(nn.Module):
    def __init__(self, in_channels, hidden_channels=32, out_channels=1, heads=2, dropout=0.6):
        """
        Graph Attention Network with strong regularization
        
        Args:
            in_channels (int): Number of input features
            hidden_channels (int): Number of hidden units
            out_channels (int): Number of output classes
            heads (int): Number of attention heads
            dropout (float): Dropout rate
        """
        super(SSGAT, self).__init__()
        
        # Feature preprocessing with strong regularization
        self.preprocess = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, hidden_channels),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # GAT layer with multiple heads
        self.gat = GATConv(
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            heads=heads,
            dropout=dropout,
            concat=True
        )
        
        # Layer normalization
        self.norm = nn.LayerNorm(hidden_channels * heads)
        
        # Output layer with strong regularization
        self.out = nn.Sequential(
            nn.Linear(hidden_channels * heads, hidden_channels),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.BatchNorm1d(hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, out_channels)
        )
        
        logger.info(f"Initialized Regularized SSGAT model:")
        logger.info(f"Input dimension: {in_channels}")
        logger.info(f"Hidden dimension: {hidden_channels}")
        logger.info(f"Number of attention heads: {heads}")
        logger.info(f"Dropout rate: {dropout}")
    
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
        # Feature preprocessing
        x = self.preprocess(x)
        
        # GAT layer with multi-head attention
        x = self.gat(x, edge_index, edge_weights)
        x = self.norm(x)
        x = F.elu(x)
        
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