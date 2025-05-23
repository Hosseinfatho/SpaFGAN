import zarr
import numpy as np
from typing import Dict, Any
import json
import s3fs

def get_s3fs():
    """
    Create an S3 filesystem object.
    For public data, we don't need credentials.
    """
    return s3fs.S3FileSystem(anon=True)

def analyze_zarr_structure(url: str) -> Dict[str, Any]:
    """
    Analyze the structure of a Zarr array or group.
    
    Args:
        url: URL or path to the Zarr data
        
    Returns:
        Dictionary containing the structure information
    """
    try:
        # Handle S3 paths
        if url.startswith('s3://'):
            fs = get_s3fs()
            store = s3fs.S3Map(root=url, s3=fs)
            z = zarr.open(store, mode='r')
        else:
            z = zarr.open(url, mode='r')
        
        # Initialize result dictionary
        result = {
            'type': 'group' if isinstance(z, zarr.hierarchy.Group) else 'array',
            'structure': {}
        }
        
        if isinstance(z, zarr.hierarchy.Group):
            # If it's a group, analyze its contents
            for key, item in z.items():
                if isinstance(item, zarr.core.Array):
                    result['structure'][key] = {
                        'type': 'array',
                        'shape': item.shape,
                        'dtype': str(item.dtype),
                        'chunks': item.chunks,
                        'compressor': str(item.compressor),
                        'size': item.nbytes / (1024 * 1024 * 1024)  # Size in GB
                    }
                elif isinstance(item, zarr.hierarchy.Group):
                    result['structure'][key] = {
                        'type': 'group',
                        'contents': analyze_zarr_structure(f"{url}/{key}")['structure']
                    }
        else:
            # If it's an array, get its properties
            result['structure'] = {
                'shape': z.shape,
                'dtype': str(z.dtype),
                'chunks': z.chunks,
                'compressor': str(z.compressor),
                'size': z.nbytes / (1024 * 1024 * 1024)  # Size in GB
            }
            
        return result
    
    except Exception as e:
        return {'error': str(e)}

def print_zarr_tree(url: str, indent: int = 0) -> None:
    """
    Print a tree representation of the Zarr structure.
    
    Args:
        url: URL or path to the Zarr data
        indent: Current indentation level
    """
    try:
        # Handle S3 paths
        if url.startswith('s3://'):
            fs = get_s3fs()
            store = s3fs.S3Map(root=url, s3=fs)
            z = zarr.open(store, mode='r')
        else:
            z = zarr.open(url, mode='r')
        
        if isinstance(z, zarr.hierarchy.Group):
            for key, item in z.items():
                print(' ' * indent + f'├── {key}')
                if isinstance(item, zarr.hierarchy.Group):
                    print_zarr_tree(f"{url}/{key}", indent + 4)
                else:
                    size_gb = item.nbytes / (1024 * 1024 * 1024)
                    print(' ' * (indent + 4) + f'├── shape: {item.shape}')
                    print(' ' * (indent + 4) + f'├── dtype: {item.dtype}')
                    print(' ' * (indent + 4) + f'└── size: {size_gb:.2f} GB')
        else:
            size_gb = z.nbytes / (1024 * 1024 * 1024)
            print(' ' * indent + f'├── shape: {z.shape}')
            print(' ' * indent + f'├── dtype: {z.dtype}')
            print(' ' * indent + f'└── size: {size_gb:.2f} GB')
            
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    # Example usage
    url = "s3://lsp-public-data/yapp-2023-3d-melanoma/Dataset1-LSP13626-melanoma-in-situ/0/3"
    #url = "D:/Research/vis2025/BestCameraPosition/backend/Input/default_channels_20250521_095553.zarr"  # Using forward slashes for Windows path

    print("Analyzing Zarr structure...")
    structure = analyze_zarr_structure(url)
    print("\nDetailed structure:")
    print(json.dumps(structure, indent=2))
    
    print("\nTree representation:")
    print_zarr_tree(url) 