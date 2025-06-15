import pandas as pd
import numpy as np
from pathlib import Path
import logging
from sklearn.preprocessing import MinMaxScaler
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define interaction logic
INTERACTIONS = {
    "T-cell entry site": {"CD31": "high", "CD4": "high"},
    "Inflammatory zone": {"CD11b": "high", "CD20": "high"},
    "Oxidative stress niche": {"CD11b": "high", "Catalase": "high"},
    "B-cell infiltration": {"CD20": "high", "CD31": "high"},
    "Dendritic signal": {"CD11c": "high"}
}

def calculate_global_stats(df):
    """Calculate global statistics for each marker"""
    stats = {}
    for marker in ['CD31', 'CD4', 'CD20', 'CD11b', 'CD11c', 'Catalase']:
        if marker in df.columns:
            values = df[marker].values
            stats[marker] = {
                'min': np.percentile(values, 5),
                'max': np.percentile(values, 95),
                'mean': np.mean(values),
                'std': np.std(values)
            }
    return stats

def normalize(value, marker, df):
    """Normalize using different methods based on marker type"""
    marker_values = df[marker].values
    
    # For single-marker interactions (like Dendritic signal), use simple normalization
    if marker == 'CD11c':  # Dendritic signal marker
        max_value = np.max(marker_values)
        if max_value == 0:
            return 0.0
        return round(value / max_value, 4)
    
    # For other markers, use z-score and sigmoid
    mean = np.mean(marker_values)
    std = np.std(marker_values)
    
    if std == 0:
        return 0.5
    
    # Calculate z-score
    z_score = (value - mean) / std
    
    # Apply sigmoid function to get value between 0 and 1
    sigmoid = 1 / (1 + np.exp(-z_score))
    
    # Round to 4 decimal places
    return round(sigmoid, 4)

def calculate_interaction_score(row, df, interaction_name):
    """Calculate normalized score for a specific interaction"""
    markers = INTERACTIONS[interaction_name]
    scores = []
    
    # Special case for Dendritic signal (single marker)
    if interaction_name == "Dendritic signal":
        if 'CD11c' in df.columns:
            value = row['CD11c']
            return normalize(value, 'CD11c', df)
        return 0.0
    
    # For other interactions
    for marker, threshold in markers.items():
        if marker in df.columns:
            value = row[marker]
            norm_value = normalize(value, marker, df)
            scores.append(norm_value)
    
    if scores:
        # Use minimum of scores to be more strict
        return round(min(scores), 4)
    return 0.0

def plot_rois(roi_df, output_dir, marker):
    """
    Plot ROIs with color intensity based on interaction scores
    
    Args:
        roi_df (pd.DataFrame): DataFrame containing ROI data
        output_dir (Path): Directory to save plot
        marker (str): Current marker being processed
    """
    fig, ax = plt.subplots(figsize=(15, 10))
    
    # Create a colormap for interactions
    interaction_colors = {
        "T-cell entry site": "red",
        "Inflammatory zone": "blue",
        "Oxidative stress niche": "green",
        "B-cell infiltration": "purple",
        "Dendritic signal": "orange"
    }
    
    # Plot each cell with color based on its interactions
    for _, row in roi_df.iterrows():
        interactions = row['interactions'].split(',')
        # Use the first interaction for color (you can modify this logic)
        color = interaction_colors[interactions[0]]
        # Use max_interaction_score for alpha
        alpha = row['max_interaction_score']
        ax.scatter(row['x'], row['y'], 
                  c=[color], 
                  alpha=alpha,
                  s=50,  # size of points
                  edgecolors='black',
                  linewidths=0.5)
    
    # Add legend
    legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                                 markerfacecolor=color, label=name, markersize=10)
                      for name, color in interaction_colors.items()]
    ax.legend(handles=legend_elements, loc='upper right')
    
    # Add colorbar for intensity
    sm = plt.cm.ScalarMappable(cmap=plt.cm.Greys, norm=plt.Normalize(0, 1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label('Interaction Score')
    
    ax.set_title(f"ROI Plot - {marker}")
    ax.set_xlabel("X Position")
    ax.set_ylabel("Y Position")
    
    # Save plot
    plot_path = output_dir / f"roi_plot_{marker}.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved ROI plot to: {plot_path}")

def extract_rois(cell_features_path, output_dir, marker):
    """
    Extract ROIs based on cell features and interactions
    
    Args:
        cell_features_path (Path): Path to cell features CSV
        output_dir (Path): Directory to save output files
        marker (str): Current marker being processed
    """
    logger.info(f"Extracting ROIs for {marker}...")
    
    # Load cell features
    df = pd.read_csv(cell_features_path)
    logger.info(f"Loaded {len(df)} cells")
    
    # Calculate interaction scores for each cell
    roi_scores = []
    
    for _, row in df.iterrows():
        interaction_scores = {}
        active_interactions = []
        
        # Calculate score for each interaction
        for interaction_name in INTERACTIONS.keys():
            score = calculate_interaction_score(row, df, interaction_name)
            interaction_scores[interaction_name] = score
            
            # Check if interaction is active (score > 0.3)
            if score > 0.3:
                active_interactions.append(interaction_name)
        
        if active_interactions:
            roi_scores.append({
                'cell_id': row['cell_id'],
                'x': row['x'],
                'y': row['y'],
                'z': row['z'],
                'interactions': ','.join(active_interactions),
                **interaction_scores  # Add all interaction scores
            })
    
    # Create ROI dataframe
    roi_df = pd.DataFrame(roi_scores)
    
    if len(roi_df) > 0:
        # Sort by highest interaction score
        interaction_columns = list(INTERACTIONS.keys())
        roi_df['max_interaction_score'] = roi_df[interaction_columns].max(axis=1)
        roi_df = roi_df.sort_values('max_interaction_score', ascending=False)
        
        # Save ROIs
        output_path = output_dir / f"extracted_rois_{marker}.csv"
        roi_df.to_csv(output_path, index=False)
        logger.info(f"Saved {len(roi_df)} ROIs to {output_path}")
        
        # Plot ROIs
        plot_rois(roi_df, output_dir, marker)
        
        # Print statistics
        logger.info("\nROI Statistics:")
        logger.info(f"Total cells: {len(df)}")
        logger.info(f"ROI cells: {len(roi_df)}")
        logger.info(f"ROI percentage: {(len(roi_df)/len(df)*100):.1f}%")
        
        # Interaction statistics
        logger.info("\nInteraction Statistics:")
        for interaction in INTERACTIONS.keys():
            count = len(roi_df[roi_df[interaction] > 0.3])
            avg_score = roi_df[roi_df[interaction] > 0.3][interaction].mean()
            logger.info(f"{interaction}: {count} cells (avg score: {avg_score:.3f})")
    else:
        logger.warning(f"No ROIs found for {marker}")

def plot_all_markers_rois(output_dir):
    """
    Plot ROIs for all markers in one plot, showing top 30% interactions
    with different shapes for each marker
    """
    fig, ax = plt.subplots(figsize=(20, 15))
    
    # Define marker shapes and colors
    marker_shapes = {
        "CD31": "o",  # Circle
        "CD11b": "s",  # Square
        "CD11c": "^"   # Triangle
    }
    
    interaction_colors = {
        "T-cell entry site": "red",
        "Inflammatory zone": "blue",
        "Oxidative stress niche": "green",
        "B-cell infiltration": "purple",
        "Dendritic signal": "orange"
    }
    
    # Process each marker
    for marker in ["CD31", "CD11b", "CD11c"]:
        roi_path = output_dir / f"extracted_rois_{marker}.csv"
        if not roi_path.exists():
            logger.warning(f"ROI file not found for {marker}, skipping...")
            continue
            
        # Load ROI data
        roi_df = pd.read_csv(roi_path)
        
        # Get top 30% for each interaction
        for interaction in INTERACTIONS.keys():
            if interaction in roi_df.columns:
                # Get cells with this interaction
                interaction_cells = roi_df[roi_df[interaction] > 0.3].copy()
                if len(interaction_cells) > 0:
                    # Sort by interaction score and get top 30%
                    n_top = max(1, int(len(interaction_cells) * 0.3))  # At least 1 cell
                    top_cells = interaction_cells.nlargest(n_top, interaction)
                    
                    if len(top_cells) > 0:  # Check if we have any cells to plot
                        # Plot cells
                        ax.scatter(
                            top_cells['x'], 
                            top_cells['y'],
                            marker=marker_shapes[marker],
                            c=[interaction_colors[interaction]],
                            alpha=top_cells[interaction].values,  # Convert to numpy array
                            s=100,  # size of points
                            edgecolors='black',
                            linewidths=0.5,
                            label=f"{marker} - {interaction}"
                        )
    
    # Add legend with larger font size
    legend = ax.legend(
        bbox_to_anchor=(1.05, 1), 
        loc='upper left',
        fontsize=20,  # Increased font size
        title="Markers and Interactions",
        title_fontsize=24  # Title font size
    )
    
    ax.set_title("ROI Plot - All Markers (Top 30% Interactions)", fontsize=24)
    ax.set_xlabel("X Position", fontsize=20)
    ax.set_ylabel("Y Position", fontsize=20)
    
    # Adjust tick label sizes
    ax.tick_params(axis='both', which='major', labelsize=16)
    
    # Save plot
    plot_path = output_dir / "roi_plot_all_markers.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved combined ROI plot to: {plot_path}")

def main():
    backend_dir = Path(__file__).parent.resolve()
    output_dir = backend_dir / "output"
    
    # Process each marker
    for marker in ["CD31", "CD11b", "CD11c"]:
        feature_path = output_dir / f"cell_features_{marker}.csv"
        if feature_path.exists():
            logger.info(f"\nProcessing {marker}...")
            extract_rois(feature_path, output_dir, marker)
        else:
            logger.warning(f"Feature file not found for {marker}, skipping...")
    
    # Create combined plot
    plot_all_markers_rois(output_dir)

if __name__ == "__main__":
    main()
