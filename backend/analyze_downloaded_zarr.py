import zarr
import numpy as np
import json
import pandas as pd
from pathlib import Path

def analyze_zarr_structure():
    """
    Analyze the downloaded Zarr file structure
    """
    
    # Open the zarr file
    zarr_path = "../Globally Sclerotic Glomeruli.adata.zarr"
    root = zarr.open_group(zarr_path, mode='r')
    
    print("=== Zarr File Structure Analysis ===")
    print(f"Zarr path: {zarr_path}")
    print()
    
    # Check main attributes
    print("=== Main Attributes ===")
    if '.zattrs' in root:
        attrs = root.attrs
        print(f"Main attributes: {dict(attrs)}")
    print()
    
    # Analyze obs (observations)
    print("=== Observations (obs) ===")
    if 'obs' in root:
        obs_group = root['obs']
        print(f"obs attributes: {dict(obs_group.attrs)}")
        
        # List all obs datasets
        print("obs datasets:")
        for key in obs_group.keys():
            if key != '_index':
                dataset = obs_group[key]
                print(f"  - {key}: shape={dataset.shape}, dtype={dataset.dtype}")
                if hasattr(dataset, 'attrs'):
                    print(f"    attributes: {dict(dataset.attrs)}")
        
        # Get obs index
        if '_index' in obs_group:
            obs_index = obs_group['_index'][:]
            print(f"obs index shape: {obs_index.shape}")
            print(f"obs index: {obs_index}")
    print()
    
    # Analyze var (variables)
    print("=== Variables (var) ===")
    if 'var' in root:
        var_group = root['var']
        print(f"var attributes: {dict(var_group.attrs)}")
        
        # List all var datasets
        print("var datasets:")
        for key in var_group.keys():
            if key != '_index':
                dataset = var_group[key]
                print(f"  - {key}: shape={dataset.shape}, dtype={dataset.dtype}")
                if hasattr(dataset, 'attrs'):
                    print(f"    attributes: {dict(dataset.attrs)}")
        
        # Get var index
        if '_index' in var_group:
            var_index = var_group['_index'][:]
            print(f"var index shape: {var_index.shape}")
            print(f"var index: {var_index}")
    print()
    
    # Analyze X (main data matrix)
    print("=== Main Data Matrix (X) ===")
    if 'X' in root:
        X_group = root['X']
        print(f"X attributes: {dict(X_group.attrs)}")
        
        # Get X data
        X_data = X_group[:]
        print(f"X shape: {X_data.shape}")
        print(f"X dtype: {X_data.dtype}")
        print(f"X data:\n{X_data}")
    print()
    
    # Analyze uns (unsorted)
    print("=== Unsorted Data (uns) ===")
    if 'uns' in root:
        uns_group = root['uns']
        print(f"uns attributes: {dict(uns_group.attrs)}")
        
        # List all uns datasets
        print("uns datasets:")
        for key in uns_group.keys():
            dataset = uns_group[key]
            print(f"  - {key}: shape={dataset.shape}, dtype={dataset.dtype}")
            if hasattr(dataset, 'attrs'):
                print(f"    attributes: {dict(dataset.attrs)}")
    print()
    
    # Analyze obsm (observation matrices)
    print("=== Observation Matrices (obsm) ===")
    if 'obsm' in root:
        obsm_group = root['obsm']
        print(f"obsm attributes: {dict(obsm_group.attrs)}")
        
        # List all obsm datasets
        print("obsm datasets:")
        for key in obsm_group.keys():
            dataset = obsm_group[key]
            print(f"  - {key}: shape={dataset.shape}, dtype={dataset.dtype}")
            if hasattr(dataset, 'attrs'):
                print(f"    attributes: {dict(dataset.attrs)}")
    print()
    
    # Analyze varm (variable matrices)
    print("=== Variable Matrices (varm) ===")
    if 'varm' in root:
        varm_group = root['varm']
        print(f"varm attributes: {dict(varm_group.attrs)}")
        
        # List all varm datasets
        print("varm datasets:")
        for key in varm_group.keys():
            dataset = varm_group[key]
            print(f"  - {key}: shape={dataset.shape}, dtype={dataset.dtype}")
            if hasattr(dataset, 'attrs'):
                print(f"    attributes: {dict(dataset.attrs)}")
    print()
    
    # Analyze obsp (observation pairs)
    print("=== Observation Pairs (obsp) ===")
    if 'obsp' in root:
        obsp_group = root['obsp']
        print(f"obsp attributes: {dict(obsp_group.attrs)}")
        
        # List all obsp datasets
        print("obsp datasets:")
        for key in obsp_group.keys():
            dataset = obsp_group[key]
            print(f"  - {key}: shape={dataset.shape}, dtype={dataset.dtype}")
            if hasattr(dataset, 'attrs'):
                print(f"    attributes: {dict(dataset.attrs)}")
    print()
    
    # Analyze varp (variable pairs)
    print("=== Variable Pairs (varp) ===")
    if 'varp' in root:
        varp_group = root['varp']
        print(f"varp attributes: {dict(varp_group.attrs)}")
        
        # List all varp datasets
        print("varp datasets:")
        for key in varp_group.keys():
            dataset = varp_group[key]
            print(f"  - {key}: shape={dataset.shape}, dtype={dataset.dtype}")
            if hasattr(dataset, 'attrs'):
                print(f"    attributes: {dict(dataset.attrs)}")
    print()
    
    # Analyze layers
    print("=== Layers ===")
    if 'layers' in root:
        layers_group = root['layers']
        print(f"layers attributes: {dict(layers_group.attrs)}")
        
        # List all layers datasets
        print("layers datasets:")
        for key in layers_group.keys():
            dataset = layers_group[key]
            print(f"  - {key}: shape={dataset.shape}, dtype={dataset.dtype}")
            if hasattr(dataset, 'attrs'):
                print(f"    attributes: {dict(dataset.attrs)}")
    print()
    
    # Create a summary
    print("=== Summary ===")
    print(f"Total observations: {X_data.shape[0] if 'X' in root else 'Unknown'}")
    print(f"Total variables: {X_data.shape[1] if 'X' in root else 'Unknown'}")
    print(f"Data type: {X_data.dtype if 'X' in root else 'Unknown'}")
    
    # Save detailed analysis to file
    analysis_result = {
        'zarr_path': zarr_path,
        'main_attributes': dict(root.attrs) if hasattr(root, 'attrs') else {},
        'structure': {
            'obs': {
                'attributes': dict(obs_group.attrs) if 'obs' in root else {},
                'datasets': {key: {'shape': dataset.shape, 'dtype': str(dataset.dtype)} 
                           for key, dataset in obs_group.items() if key != '_index'} if 'obs' in root else {}
            },
            'var': {
                'attributes': dict(var_group.attrs) if 'var' in root else {},
                'datasets': {key: {'shape': dataset.shape, 'dtype': str(dataset.dtype)} 
                           for key, dataset in var_group.items() if key != '_index'} if 'var' in root else {}
            },
            'X': {
                'shape': X_data.shape if 'X' in root else None,
                'dtype': str(X_data.dtype) if 'X' in root else None,
                'attributes': dict(X_group.attrs) if 'X' in root else {}
            }
        }
    }
    
    output_file = Path('output/zarr_analysis.json')
    output_file.parent.mkdir(exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(analysis_result, f, indent=2, default=str)
    
    print(f"Detailed analysis saved to: {output_file}")

if __name__ == "__main__":
    analyze_zarr_structure() 