import pandas as pd
import numpy as np
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define markers
MARKERS = ["CD31", "CD20", "CD11b", "CD4", "CD11c", "Catalase"]

def normalize_marker_values(df):
    """Normalize marker values to [0,1] range"""
    normalized_df = df.copy()
    
    # Normalize each marker column
    for marker in MARKERS:
        if marker in df.columns:
            max_val = df[marker].max()
            if max_val > 0:  # Avoid division by zero
                normalized_df[marker] = df[marker] / max_val
            else:
                normalized_df[marker] = 0
            logger.info(f"Normalized {marker}: max value was {max_val:.2f}")
    
    return normalized_df

def merge_cell_features():
    """Merge cell features from all markers into one file with marker tags"""
    all_cells = []
    
    # Process each marker's cell features
    for marker in MARKERS:
        file_path = f"output/cell_features_{marker}.csv"
        if Path(file_path).exists():
            # Read marker's cell features
            df = pd.read_csv(file_path)
            logger.info(f"Loaded {len(df)} cells from {marker}")
            
            # Add marker tag
            df['marker'] = marker
            
            # Add to combined list
            all_cells.append(df)
        else:
            logger.warning(f"File not found: {file_path}")
    
    if not all_cells:
        logger.error("No cell feature files found!")
        return
    
    # Combine all cell features
    combined_df = pd.concat(all_cells, ignore_index=True)
    
    # Save non-normalized features
    output_path = "output/All_cell_features.csv"
    combined_df.to_csv(output_path, index=False)
    logger.info(f"Saved {len(combined_df)} total cells to {output_path}")
    
    # Normalize marker values
    normalized_df = normalize_marker_values(combined_df)
    
    # Save normalized features
    norm_output_path = "output/Norm_all_cell_features.csv"
    normalized_df.to_csv(norm_output_path, index=False)
    logger.info(f"Saved normalized features to {norm_output_path}")
    
    # Log statistics
    logger.info("\nCell counts by marker:")
    for marker in MARKERS:
        marker_cells = combined_df[combined_df['marker'] == marker]
        logger.info(f"{marker}: {len(marker_cells)} cells")
    
    # Log normalization statistics
    logger.info("\nNormalization statistics:")
    for marker in MARKERS:
        if marker in combined_df.columns:
            orig_max = combined_df[marker].max()
            norm_max = normalized_df[marker].max()
            logger.info(f"{marker}: Original max = {orig_max:.2f}, Normalized max = {norm_max:.2f}")

if __name__ == "__main__":
    merge_cell_features() 