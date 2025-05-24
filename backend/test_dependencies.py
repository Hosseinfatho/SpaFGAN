import torch
import torch_geometric
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from flask import Flask
from flask_cors import CORS
import zarr
import networkx as nx
from skimage import io
from tqdm import tqdm

def test_dependencies():
    print("Testing dependencies...")
    
    # Test PyTorch and CUDA
    print("\nPyTorch and CUDA:")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    
    # Test NumPy and Pandas
    print("\nNumPy and Pandas:")
    arr = np.random.rand(5, 5)
    df = pd.DataFrame(arr)
    print("NumPy array shape:", arr.shape)
    print("Pandas DataFrame shape:", df.shape)
    
    # Test Matplotlib and Seaborn
    print("\nMatplotlib and Seaborn:")
    plt.figure(figsize=(5, 5))
    sns.heatmap(arr)
    plt.close()
    print("Plot created successfully")
    
    # Test Flask
    print("\nFlask:")
    app = Flask(__name__)
    CORS(app)
    print("Flask app created successfully")
    
    # Test NetworkX
    print("\nNetworkX:")
    G = nx.Graph()
    G.add_edge(1, 2)
    print("Graph created with 2 nodes and 1 edge")
    
    # Test tqdm
    print("\ntqdm:")
    for _ in tqdm(range(3), desc="Testing tqdm"):
        pass
    
    print("\nAll dependencies tested successfully!")

if __name__ == "__main__":
    test_dependencies() 