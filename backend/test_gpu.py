import torch
import torch_geometric
from torch_geometric.nn import GATConv

def test_gpu():
    print("PyTorch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("CUDA version:", torch.version.cuda)
        print("GPU device:", torch.cuda.get_device_name(0))
    
    # Create a simple graph
    x = torch.randn(4, 16)  # 4 nodes with 16 features each
    edge_index = torch.tensor([[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]], dtype=torch.long)
    
    # Move to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x = x.to(device)
    edge_index = edge_index.to(device)
    
    # Create a GAT layer
    conv = GATConv(16, 8).to(device)
    
    # Forward pass
    out = conv(x, edge_index)
    print("\nOutput shape:", out.shape)
    print("Device of output:", out.device)
    print("\nTest completed successfully!")

if __name__ == "__main__":
    test_gpu() 