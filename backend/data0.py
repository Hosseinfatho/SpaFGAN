import zarr
import numpy as np
from typing import Dict, Any, List, Tuple
import json
import s3fs
import requests
import ome_types

def get_s3fs():
    """
    Create an S3 filesystem object.
    For public data, we don't need credentials.
    """
    return s3fs.S3FileSystem(anon=True)

def get_channel_names_from_ome(url: str) -> List[str]:
    """
    Get channel names from OME-XML metadata.
    
    Args:
        url: Base URL of the dataset
        
    Returns:
        List of channel names
    """
    try:
        # Construct the OME-XML metadata URL
        metadata_url = url.replace("/0/3", "/OME/METADATA.ome.xml")
        
        # Get the OME-XML data
        response = requests.get(metadata_url)
        data = response.text
        
        # Parse the OME-XML
        ome_xml = ome_types.from_xml(data.replace("Â",""))
        
        # Extract channel names
        channel_names = [c.name for c in ome_xml.images[0].pixels.channels]
        
        return channel_names
    except Exception as e:
        print(f"Error getting channel names from OME-XML: {str(e)}")
        return []

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

def get_channel_info(url: str) -> List[Tuple[int, str]]:
    """
    Get channel names and their indices from the Zarr data.
    
    Args:
        url: URL or path to the Zarr data
        
    Returns:
        List of tuples containing (index, channel_name)
    """
    try:
        # First try to get channel names from OME-XML
        channel_names = get_channel_names_from_ome(url)
        
        if channel_names:
            return [(idx, name) for idx, name in enumerate(channel_names)]
        
        # If OME-XML method fails, fall back to Zarr method
        if url.startswith('s3://'):
            fs = get_s3fs()
            store = s3fs.S3Map(root=url, s3=fs)
            z = zarr.open(store, mode='r')
        else:
            z = zarr.open(url, mode='r')
        
        channel_info = []
        
        if isinstance(z, zarr.hierarchy.Group):
            # Look for channel names in group attributes
            if hasattr(z, 'attrs') and 'channel_names' in z.attrs:
                channel_names = z.attrs['channel_names']
                for idx, name in enumerate(channel_names):
                    channel_info.append((idx, name))
            else:
                # If no channel names found, use array names as channels
                for idx, (key, item) in enumerate(z.items()):
                    if isinstance(item, zarr.core.Array):
                        channel_info.append((idx, key))
        else:
            # If it's a single array, check its attributes
            if hasattr(z, 'attrs') and 'channel_names' in z.attrs:
                channel_names = z.attrs['channel_names']
                for idx, name in enumerate(channel_names):
                    channel_info.append((idx, name))
            else:
                # If no channel names found, use default naming
                channel_info.append((0, 'channel_0'))
        
        return channel_info
    
    except Exception as e:
        print(f"Error getting channel info: {str(e)}")
        return []

def get_channel_index(channel_name: str, channel_list: List[str]) -> int:
    """
    Get the index of a specific channel in the channel list.
    
    Args:
        channel_name: Name of the channel to find
        channel_list: List of all channel names
        
    Returns:
        Index of the channel (0-based) or -1 if not found
    """
    try:
        return channel_list.index(channel_name)
    except ValueError:
        return -1

if __name__ == "__main__":
    # Example usage
    url = "s3://lsp-public-data/biomedvis-challenge-2025/Dataset1-LSP13626-melanoma-in-situ/0"
    #url = "D:/Research/vis2025/BestCameraPosition/backend/Input/default_channels_20250521_095553.zarr"  # Using forward slashes for Windows path

    print("Analyzing Zarr structure...")
    structure = analyze_zarr_structure(url)
    print("\nDetailed structure:")
    print(json.dumps(structure, indent=2))
    
    print("\nTree representation:")
    print_zarr_tree(url)
    
    print("\nChannel Information:")
    channel_info = get_channel_info(url)
    channel_names = [name for _, name in channel_info]
    
    # Find CD11c channel index
    cd11c_index = get_channel_index('CD11c', channel_names)
    if cd11c_index != -1:
        print(f"\nCD11c channel index: {cd11c_index}")
    else:
        print("\nCD11c channel not found") 