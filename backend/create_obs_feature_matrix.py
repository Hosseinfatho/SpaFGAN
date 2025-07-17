import numpy as np
import zarr
import json
import os
import pandas as pd
from anndata import AnnData

def create_obs_feature_matrix():
    """
    Create 4 separate obsFeatureMatrix.anndata.zarr files, one for each interaction.
    Save them in output/vitessce_files directory.
    """
    # Read ROI data from JSON files
    interaction_files = [
        "output/top_roi_scores_B-cell_infiltration.json",
        "output/top_roi_scores_Inflammatory_zone.json",
        "output/top_roi_scores_Oxidative_stress_niche.json",
        "output/top_roi_scores_T-cell_entry_site.json"
    ]
    interaction_names = [
        "B-cell_infiltration",
        "Inflammatory_zone",
        "Oxidative_stress_niche",
        "T-cell_entry_site"
    ]
    
    # Create output directory
    output_dir = "output/vitessce_files"
    os.makedirs(output_dir, exist_ok=True)
    
    # Process each interaction separately
    for interaction_name, json_file in zip(interaction_names, interaction_files):
        print(f"\nProcessing {interaction_name}...")
        
        if not os.path.exists(json_file):
            print(f"Warning: File {json_file} not found, skipping...")
            continue
            
        with open(json_file, 'r') as f:
            interaction_data = json.load(f)
        
        rois = interaction_data.get('top_rois', [])
        print(f"Found {len(rois)} ROIs for {interaction_name}")
        
        if len(rois) == 0:
            print(f"No ROIs found for {interaction_name}, skipping...")
            continue
        
        # Create feature matrix (X) - one row per ROI, one column per interaction
        n_rois = len(rois)
        n_features = len(interaction_names)
        
        # Initialize feature matrix with zeros
        X = np.zeros((n_rois, n_features), dtype=np.float32)
        
        # Create obs (observations) dataframe with scores
        obs_data = {
            'roi_id': [],
            'interaction': [],
            'combined_score': [],
            'intensity_score': [],
            'attention_score': [],
            'num_nodes': [],
            'num_edges': []
        }
        
        # Fill data for this interaction
        for i, roi in enumerate(rois):
            roi_id = f"{interaction_name}_{roi['roi_id']}"
            scores = roi['scores']
            
            # Set the feature value for this interaction
            interaction_idx = interaction_names.index(interaction_name)
            X[i, interaction_idx] = scores['combined_score']
            
            # Add observation data
            obs_data['roi_id'].append(roi_id)
            obs_data['interaction'].append(interaction_name)
            obs_data['combined_score'].append(scores['combined_score'])
            obs_data['intensity_score'].append(scores['intensity_score'])
            obs_data['attention_score'].append(scores['attention_score'])
            obs_data['num_nodes'].append(roi.get('num_nodes', 0))
            obs_data['num_edges'].append(roi.get('num_edges', 0))
        
        # Create var (variables) dataframe
        var_data = {
            'feature_name': interaction_names,
            'feature_type': ['interaction'] * len(interaction_names)
        }
        
        # Create AnnData object
        adata = AnnData(
            X=X,
            obs=pd.DataFrame(obs_data),
            var=pd.DataFrame(var_data),
            dtype=np.float32
        )
        
        # Add pre-normalized layer for better performance
        try:
            from vitessce.data_utils import to_uint8
            adata.layers['X_uint8'] = to_uint8(adata.X, norm_along="global")
            print(f"Added pre-normalized layer X_uint8 for {interaction_name}")
        except ImportError:
            print(f"vitessce.data_utils not available, skipping pre-normalized layer for {interaction_name}")
        
        # Save as zarr
        safe_name = interaction_name.replace(' ', '_').replace('-', '_')
        output_file = f"{output_dir}/{safe_name}_obsFeatureMatrix.anndata.zarr"
        adata.write_zarr(output_file)
        
        print(f"Created AnnData zarr file: {output_file}")
        print(f"Shape: {adata.shape}")
        print(f"Observations: {adata.n_obs}")
        print(f"Variables: {adata.n_vars}")
        print(f"Obs columns: {list(adata.obs.columns)}")
        print(f"Var columns: {list(adata.var.columns)}")
        
        # Verify the file structure
        print(f"Verifying zarr structure for {interaction_name}:")
        z = zarr.open(output_file, 'r')
        print(f"Keys: {list(z.keys())}")
        print(f"X shape: {z['X'].shape}")
        print(f"Obs keys: {list(z['obs'].keys())}")
        print(f"Var keys: {list(z['var'].keys())}")
    
    print(f"\nAll files created in: {output_dir}")
    return output_dir

if __name__ == "__main__":
    create_obs_feature_matrix() 