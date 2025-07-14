#!/usr/bin/env python3
"""
Script to inspect the structure of Zarr files used in the project
"""

import zarr
import json
from pathlib import Path
import numpy as np

def inspect_zarr_structure(zarr_path, max_items=5):
    """Inspect the complete structure of a Zarr file"""
    print(f"\n{'='*60}")
    print(f"Inspecting: {zarr_path}")
    print(f"{'='*60}")
    
    try:
        # Open the Zarr group
        z = zarr.open(str(zarr_path), mode='r')
        
        def explore_group(group, path="", level=0):
            indent = "  " * level
            print(f"{indent}📁 {path or 'root'}")
            
            # Print attributes
            if hasattr(group, 'attrs') and group.attrs:
                print(f"{indent}  📋 Attributes:")
                for key, value in group.attrs.items():
                    if isinstance(value, dict):
                        print(f"{indent}    {key}: {json.dumps(value, indent=2)}")
                    else:
                        print(f"{indent}    {key}: {value}")
            
            # Explore arrays
            if hasattr(group, 'keys'):
                for key in group.keys():
                    item = group[key]
                    item_path = f"{path}/{key}" if path else key
                    
                    if hasattr(item, 'shape'):  # It's an array
                        print(f"{indent}  📊 Array: {key}")
                        print(f"{indent}    Shape: {item.shape}")
                        print(f"{indent}    Dtype: {item.dtype}")
                        print(f"{indent}    Chunks: {item.chunks}")
                        
                        # Show sample data
                        if item.size > 0:
                            try:
                                sample = item[:min(max_items, item.shape[0])]
                                if item.ndim == 1:
                                    print(f"{indent}    Sample: {sample}")
                                else:
                                    print(f"{indent}    Sample shape: {sample.shape}")
                                    print(f"{indent}    First few values: {sample.flatten()[:max_items]}")
                            except Exception as e:
                                print(f"{indent}    Error reading sample: {e}")
                        
                        # Print array attributes
                        if hasattr(item, 'attrs') and item.attrs:
                            print(f"{indent}    Array attributes:")
                            for attr_key, attr_value in item.attrs.items():
                                print(f"{indent}      {attr_key}: {attr_value}")
                    
                    elif hasattr(item, 'keys'):  # It's a group
                        explore_group(item, item_path, level + 1)
        
        explore_group(z)
        
    except Exception as e:
        print(f"Error inspecting {zarr_path}: {e}")

def main():
    """Main function to inspect all Zarr files"""
    output_dir = Path(__file__).parent / "output"
    
    # List of Zarr files to inspect (local)
    local_zarr_files = [
        "test.zarr",
        "roi_shapes.spatialdata.zarr"
    ]
    
    # List of remote Zarr files to inspect
    remote_zarr_files = [
        "https://storage.googleapis.com/vitessce-demo-data/maynard-2021/151673.sdata.zarr"
    ]
    
    print("🔍 Zarr Structure Inspector")
    print("="*60)
    
    # Inspect local files
    print("\n📂 LOCAL FILES:")
    for zarr_file in local_zarr_files:
        zarr_path = output_dir / zarr_file
        if zarr_path.exists():
            inspect_zarr_structure(zarr_path)
        else:
            print(f"\n❌ File not found: {zarr_path}")
    
    # Inspect remote files
    print("\n🌐 REMOTE FILES:")
    for zarr_url in remote_zarr_files:
        try:
            print(f"\n{'='*60}")
            print(f"Connecting to: {zarr_url}")
            print(f"{'='*60}")
            inspect_zarr_structure(zarr_url)
        except Exception as e:
            print(f"❌ Error accessing remote file: {e}")
    
    print(f"\n{'='*60}")
    print("✅ Inspection complete!")

if __name__ == "__main__":
    main() 