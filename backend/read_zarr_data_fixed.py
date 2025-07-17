import zarr
import numpy as np
import pandas as pd
import json
from pathlib import Path

def read_zarr_data():
    """
    Read actual data from the downloaded Zarr file
    """
    
    # Open the zarr file
    zarr_path = "../Globally Sclerotic Glomeruli.adata.zarr"
    root = zarr.open_group(zarr_path, mode='r')
    
    print("=== Reading Zarr Data ===")
    print(f"Zarr path: {zarr_path}")
    print()
    
    # Read obs (observations) data
    print("=== Observations Data ===")
    obs_group = root['obs']
    
    # Read obs index
    obs_index = obs_group['_index'][:]
    print(f"Observation names: {obs_index}")
    print(f"Number of observations: {len(obs_index)}")
    print()
    
    # Create unique observation names
    unique_obs_names = [f"Glomerulus_{i}" for i in range(len(obs_index))]
    print(f"Unique observation names: {unique_obs_names}")
    print()
    
    # Read obs datasets
    obs_data = {}
    for key in obs_group.keys():
        if key != '_index':
            data = obs_group[key][:]
            obs_data[key] = data
            print(f"{key}: {data}")
            print(f"  - Shape: {data.shape}")
            print(f"  - Type: {data.dtype}")
            print(f"  - Min: {data.min()}")
            print(f"  - Max: {data.max()}")
            print(f"  - Mean: {data.mean()}")
            print()
    
    # Create obs DataFrame with unique index
    obs_df = pd.DataFrame(obs_data, index=unique_obs_names)
    print("Observations DataFrame:")
    print(obs_df)
    print()
    
    # Read var (variables) data
    print("=== Variables Data ===")
    var_group = root['var']
    
    # Read var index
    var_index = var_group['_index'][:]
    print(f"Variable names: {var_index}")
    print(f"Number of variables: {len(var_index)}")
    print()
    
    # Read X (main data matrix)
    print("=== Main Data Matrix (X) ===")
    X_group = root['X']
    X_data = X_group[:]
    print(f"X shape: {X_data.shape}")
    print(f"X dtype: {X_data.dtype}")
    print("X data:")
    print(X_data)
    print()
    
    # Create X DataFrame with unique index
    X_df = pd.DataFrame(X_data, index=unique_obs_names, columns=var_index)
    print("X DataFrame:")
    print(X_df)
    print()
    
    # Verify that X matches obs data
    print("=== Verification ===")
    print("Checking if X data matches obs data...")
    
    # Compare X with obs data
    for i, var_name in enumerate(var_index):
        x_col = X_data[:, i]
        obs_col = obs_data[var_name]
        
        if np.allclose(x_col, obs_col):
            print(f"✓ {var_name}: X and obs data match")
        else:
            print(f"✗ {var_name}: X and obs data do not match")
            print(f"  X: {x_col}")
            print(f"  obs: {obs_col}")
    print()
    
    # Read uns (unsorted) data
    print("=== Unsorted Data ===")
    uns_group = root['uns']
    print(f"uns attributes: {dict(uns_group.attrs)}")
    
    uns_data = {}
    for key in uns_group.keys():
        data = uns_group[key][:]
        uns_data[key] = data
        print(f"{key}: {data}")
        print(f"  - Shape: {data.shape}")
        print(f"  - Type: {data.dtype}")
    print()
    
    # Create summary
    summary = {
        'zarr_path': zarr_path,
        'num_observations': len(obs_index),
        'num_variables': len(var_index),
        'original_observation_names': obs_index.tolist(),
        'unique_observation_names': unique_obs_names,
        'variable_names': var_index.tolist(),
        'data_shape': X_data.shape,
        'data_type': str(X_data.dtype),
        'obs_data': obs_df.to_dict('index'),
        'var_data': {},  # No var datasets in this file
        'X_data': X_data.tolist(),
        'uns_data': uns_data
    }
    
    # Save summary to file
    output_file = Path('output/zarr_data_summary.json')
    output_file.parent.mkdir(exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    print(f"Data summary saved to: {output_file}")
    
    # Save DataFrames to CSV
    obs_df.to_csv('output/obs_data.csv')
    X_df.to_csv('output/X_data.csv')
    print("DataFrames saved to CSV files:")
    print("  - output/obs_data.csv")
    print("  - output/X_data.csv")
    
    return obs_df, X_df, summary

if __name__ == "__main__":
    obs_df, X_df, summary = read_zarr_data() 