import pandas as pd
import numpy as np
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def prepare_roi_cells():
    """
    Transform interaction columns into rows for each marker.
    """
    backend_dir = Path(__file__).parent
    output_dir = backend_dir / "output"
    
    # List of markers to process
    markers = ["CD31", "CD11b", "CD11c"]
    
    # List of interaction types
    interaction_types = [
        "T-cell entry site",
        "Inflammatory zone",
        "Oxidative stress niche",
        "B-cell infiltration",
        "Dendritic signal"
    ]
    
    # Base columns to keep
    base_columns = [
        'roi_id', 'cell_id', 'x', 'y', 'z', 'spafgan_score',
        'CD31', 'CD20', 'CD11b', 'CD4', 'CD11c', 'Catalase'
    ]
    
    for marker in markers:
        # Read the ROI cells file for this marker
        roi_cells_path = output_dir / f"roi_cells_{marker}.csv"
        if not roi_cells_path.exists():
            logger.warning(f"File not found: {roi_cells_path}")
            continue
            
        df = pd.read_csv(roi_cells_path)
        
        # Create a list to store transformed dataframes
        transformed_dfs = []
        
        # Process each interaction type
        for interaction in interaction_types:
            # Select base columns and the current interaction
            interaction_df = df[base_columns + [interaction]].copy()
            
            # Rename the interaction column to 'interaction_score'
            interaction_df = interaction_df.rename(columns={interaction: 'interaction_score'})
            
            # Add interaction type column
            interaction_df['interaction_type'] = interaction
            
            # Sort by interaction score
            interaction_df = interaction_df.sort_values('interaction_score', ascending=False)
            
            # Add to transformed list
            transformed_dfs.append(interaction_df)
            
            logger.info(f"Processed {interaction} for {marker}")
            logger.info(f"Number of ROIs: {len(interaction_df)}")
            logger.info(f"Score Range: {interaction_df['interaction_score'].min():.3f} - {interaction_df['interaction_score'].max():.3f}")
            logger.info("-" * 50)
        
        # Combine all transformed dataframes
        combined_df = pd.concat(transformed_dfs, ignore_index=True)
        
        # Save combined file
        output_path = output_dir / f"roi_cells_{marker}_all.csv"
        combined_df.to_csv(output_path, index=False)
        
        logger.info(f"\nCreated combined file: {output_path}")
        logger.info(f"Total rows: {len(combined_df)}")
        logger.info(f"Unique ROIs: {combined_df['roi_id'].nunique()}")

if __name__ == "__main__":
    prepare_roi_cells() 