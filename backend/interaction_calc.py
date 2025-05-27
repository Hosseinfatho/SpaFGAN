import pandas as pd
import numpy as np
from pathlib import Path
import logging
from scipy.spatial.distance import cdist

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define markers and their interactions
INTERACTIONS = {
    "T-cell_entry_site": ["CD31", "CD4"],
    "Inflammatory_zone": ["CD11b", "CD20"],
    "Oxidative_stress_niche": ["CD11b", "Catalase"],
    "B-cell_infiltration": ["CD20", "CD31"],
    "Dendritic_signal": ["CD11c"]
}

def filter_rois_by_distance(df, min_distance=100):
    """
    Filter ROIs to ensure minimum distance between them
    
    Args:
        df (pd.DataFrame): DataFrame containing ROI data with x, y, z coordinates
        min_distance (float): Minimum distance between ROIs in voxels
        
    Returns:
        pd.DataFrame: Filtered DataFrame with ROIs that meet distance criteria
    """
    if df.empty:
        return df
    
    # Sort by score in descending order
    df = df.sort_values('spafgan_score', ascending=False)
    
    # Convert coordinates to numpy array
    coords = df[['x', 'y', 'z']].values
    
    # Calculate pairwise distances
    distances = cdist(coords, coords)
    
    # Initialize list of selected indices
    selected_indices = [0]  # Always include the first ROI (highest score)
    
    # Check each ROI against already selected ones
    for i in range(1, len(df)):
        # Check distance to all previously selected ROIs
        if all(distances[i, j] >= min_distance for j in selected_indices):
            selected_indices.append(i)
    
    # Return filtered DataFrame
    return df.iloc[selected_indices].reset_index(drop=True)

def calculate_interactions(df):
    """
    Calculate interaction scores for a DataFrame of ROI cells.
    
    Args:
        df (pd.DataFrame): DataFrame containing ROI cells data
        
    Returns:
        pd.DataFrame: DataFrame with added interaction scores
    """
    # Get available markers from the DataFrame
    available_markers = [col for col in df.columns if col in ["CD31", "CD4", "CD11b", "CD20", "CD11c", "Catalase"]]
    logger.info(f"Available markers: {available_markers}")
    
    # Get unique ROIs
    unique_rois = df['roi_id'].unique()
    
    # Calculate max values for each available marker
    max_values = {marker: df[marker].max() for marker in available_markers}
    
    # Create a copy of the DataFrame
    result_df = df.copy()
    
    # Calculate interaction scores for each ROI
    for roi_id in unique_rois:
        roi_mask = df['roi_id'] == roi_id
        roi_data = df[roi_mask]
        
        # Calculate mean values for each marker in this ROI
        roi_means = {marker: roi_data[marker].mean() for marker in available_markers}
        
        # Calculate each interaction score
        for interaction_name, markers in INTERACTIONS.items():
            # Check if all required markers are available
            if all(m in available_markers for m in markers):
                if len(markers) == 1:
                    # Single marker interaction (e.g., Dendritic signal)
                    score = roi_means[markers[0]] / max_values[markers[0]]
                else:
                    # Two marker interaction
                    score = 0.5 * sum(roi_means[m] / max_values[m] for m in markers)
                
                # Add score to all cells in this ROI
                result_df.loc[roi_mask, interaction_name] = score
            else:
                # If any required marker is missing, set score to 0
                result_df.loc[roi_mask, interaction_name] = 0
                missing_markers = [m for m in markers if m not in available_markers]
                logger.warning(f"Missing markers for {interaction_name}: {missing_markers}")
    
    # Filter ROIs by minimum distance
    result_df = filter_rois_by_distance(result_df, min_distance=100)
    logger.info(f"Filtered ROIs to ensure minimum distance of 100 voxels between them")
    
    return result_df

def process_marker_files():
    """
    Process all ROI cell files and add interaction scores.
    """
    backend_dir = Path(__file__).parent.resolve()
    output_dir = backend_dir / "output"
    
    for marker in ["CD31", "CD11b"]:
        input_file = output_dir / f"roi_cells_{marker}.csv"
        output_file = output_dir / f"roi_cells_{marker}_interactions.csv"
        
        if not input_file.exists():
            logger.warning(f"Input file not found: {input_file}")
            continue
        
        try:
            # Read the input file
            logger.info(f"Processing {input_file}")
            df = pd.read_csv(input_file)
            
            # Calculate interactions
            result_df = calculate_interactions(df)
            
            # Save the result
            result_df.to_csv(output_file, index=False)
            logger.info(f"Saved interaction scores to {output_file}")
            
        except Exception as e:
            logger.error(f"Error processing {input_file}: {str(e)}")

if __name__ == "__main__":
    process_marker_files() 