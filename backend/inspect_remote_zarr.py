#!/usr/bin/env python3
"""
Script to inspect the remote Zarr file from Google Storage
"""

import s3fs
import zarr
import json
from pathlib import Path

def inspect_remote_zarr():
    """Inspect the remote Zarr file from Google Storage"""
    url = "s3://vitessce-demo-data/maynard-2021/151673.sdata.zarr"
    
    print(f"🔍 Inspecting Remote Zarr: {url}")
    print("="*60)
    
    try:
        # Create S3 filesystem
        fs = s3fs.S3FileSystem(anon=True)
        print("✅ S3 filesystem created")
        
        # List contents
        print(f"\n📂 Listing contents of: {url}")
        try:
            contents = fs.ls(url)
            print(f"Found {len(contents)} items:")
            for item in contents:
                print(f"  - {item}")
        except Exception as e:
            print(f"❌ Error listing contents: {e}")
        
        # Try to open as Zarr
        print(f"\n📊 Opening as Zarr...")
        try:
            # Create S3Map
            s3_map = s3fs.S3Map(root=url, s3=fs, check=False)
            print("✅ S3Map created")
            
            # Open Zarr group
            z = zarr.open_consolidated(store=s3_map) if '.zmetadata' in s3_map else zarr.open_group(store=s3_map, mode='r')
            print(f"✅ Zarr group opened: {type(z)}")
            
            # Explore structure
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
                
                # Explore arrays and groups
                if hasattr(group, 'keys'):
                    for key in group.keys():
                        item = group[key]
                        item_path = f"{path}/{key}" if path else key
                        
                        if hasattr(item, 'shape'):  # It's an array
                            print(f"{indent}  📊 Array: {key}")
                            print(f"{indent}    Shape: {item.shape}")
                            print(f"{indent}    Dtype: {item.dtype}")
                            if hasattr(item, 'chunks'):
                                print(f"{indent}    Chunks: {item.chunks}")
                            
                            # Show sample data
                            if item.size > 0:
                                try:
                                    sample = item[:min(5, item.shape[0])]
                                    if item.ndim == 1:
                                        print(f"{indent}    Sample: {sample}")
                                    else:
                                        print(f"{indent}    Sample shape: {sample.shape}")
                                        print(f"{indent}    First few values: {sample.flatten()[:5]}")
                                except Exception as e:
                                    print(f"{indent}    Error reading sample: {e}")
                        
                        elif hasattr(item, 'keys'):  # It's a group
                            explore_group(item, item_path, level + 1)
            
            explore_group(z)
            
        except Exception as e:
            print(f"❌ Error opening Zarr: {e}")
            import traceback
            traceback.print_exc()
    
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    inspect_remote_zarr() 