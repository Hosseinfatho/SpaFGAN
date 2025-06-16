import pandas as pd
import numpy as np
from pathlib import Path
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.manifold import TSNE
from umap import UMAP
import torch
from sklearn.preprocessing import StandardScaler
import json
import glob

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_roi_data(output_dir):
    """
    Load ROI data from JSON files
    
    Args:
        output_dir (Path): Directory containing ROI JSON files
    
    Returns:
        pd.DataFrame: Combined ROI data
    """
    roi_files = glob.glob(str(output_dir / "all_extraction_roi_*.json"))
    all_rois = []
    
    for file_path in roi_files:
        with open(file_path, 'r') as f:
            data = json.load(f)
            interaction_name = data['interaction_name']
            for roi in data['rois']:
                roi_data = {
                    'x': roi['x'],
                    'y': roi['y'],
                    'score': roi['score'],
                    'interaction': interaction_name
                }
                all_rois.append(roi_data)
    
    return pd.DataFrame(all_rois)

def analyze_embeddings(df, interaction, output_dir):
    """
    Analyze ROI embeddings using t-SNE and UMAP visualization.
    
    Args:
        df (pd.DataFrame): DataFrame containing ROI data
        interaction (str): Current interaction being analyzed
        output_dir (Path): Directory to save output files
    """
    # Create embeddings directory
    embeddings_dir = output_dir / "embeddings" / interaction
    embeddings_dir.mkdir(parents=True, exist_ok=True)
    
    # Prepare features for embedding
    X = df[['x', 'y', 'score']].values
    X_scaled = StandardScaler().fit_transform(X)
    
    # Adjust perplexity based on dataset size
    n_samples = len(X)
    perplexity = min(5, max(2, n_samples // 4))  # Ensure perplexity is between 2 and 5, and less than n_samples
    
    # Apply t-SNE
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    tsne_embeddings = tsne.fit_transform(X_scaled)
    
    # Apply UMAP
    umap = UMAP(n_neighbors=min(15, n_samples-1), min_dist=0.1, n_components=2, random_state=42)
    umap_embeddings = umap.fit_transform(X_scaled)
    
    # Create visualizations
    for method, embeddings in [("t-SNE", tsne_embeddings), ("UMAP", umap_embeddings)]:
        plt.figure(figsize=(10, 10))
        df_vis = pd.DataFrame(embeddings, columns=["Dim1", "Dim2"])
        df_vis["Label"] = df['surrogate_label']
        
        sns.scatterplot(data=df_vis, x="Dim1", y="Dim2", hue="Label", 
                       palette={0: 'blue', 1: 'red'}, alpha=0.6)
        plt.title(f"{method} of ROI Embeddings by Surrogate Label - {interaction}")
        plt.xlabel("Dimension 1")
        plt.ylabel("Dimension 2")
        plt.legend(title="Surrogate Label")
        plt.tight_layout()
        plt.savefig(embeddings_dir / f'embedding_{method.lower()}.png')
        plt.close()
        
        # Save embeddings to file
        np.save(embeddings_dir / f'embedding_{method.lower()}.npy', embeddings)

def surrogate_labeling():
    """
    Perform surrogate labeling analysis on ROI data.
    """
    backend_dir = Path(__file__).parent
    output_dir = backend_dir / "output"
    
    # Load ROI data
    df = load_roi_data(output_dir)
    
    # List of interactions to process
    interactions = df['interaction'].unique()
    
    for interaction in interactions:
        # Filter data for current interaction
        interaction_df = df[df['interaction'] == interaction].copy()
        
        # Step 1: Assign surrogate labels
        # Sort scores and assign labels
        sorted_scores = interaction_df['score'].sort_values(ascending=False)
        n = len(sorted_scores)
        
        logger.info(f"Processing {interaction} with {n} samples")
        
        # Check if we have enough samples
        if n < 4:  # Need at least 2 samples for each class
            logger.warning(f"Not enough samples for {interaction} (n={n}). Skipping...")
            continue
            
        # Calculate number of samples for each class
        top_n = max(2, int(0.1 * n))  # At least 2 samples
        bottom_n = max(2, int(0.1 * n))  # At least 2 samples
        
        # Ensure we have enough samples for both classes
        if top_n + bottom_n > n:
            top_n = n // 2
            bottom_n = n - top_n
            
        logger.info(f"Assigning {top_n} samples to class 1 and {bottom_n} samples to class 0")
        
        # Assign labels
        labels = np.full(n, -1, dtype=int)
        labels[:top_n] = 1
        labels[-bottom_n:] = 0
        interaction_df['surrogate_label'] = labels
        
        # Check if we have samples from both classes
        unique_labels = interaction_df['surrogate_label'].unique()
        if len(unique_labels) < 2:
            logger.warning(f"Only one class found for {interaction}. Skipping...")
            continue
            
        # Filter out unlabeled samples
        interaction_df = interaction_df[interaction_df['surrogate_label'] != -1].reset_index(drop=True)
        
        # Verify class distribution
        class_counts = interaction_df['surrogate_label'].value_counts()
        logger.info(f"Class distribution: {class_counts.to_dict()}")
        
        if len(class_counts) < 2:
            logger.warning(f"Not enough classes after filtering for {interaction}. Skipping...")
            continue

        # Step 2: Create visualizations
        # Create output directory for plots
        plots_dir = output_dir / "surrogate_plots" / interaction
        plots_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. XY Spatial Scatter Plot
        plt.figure(figsize=(10, 10))
        sns.scatterplot(data=interaction_df, x='x', y='y', hue='surrogate_label', 
                       palette={0: 'blue', 1: 'red'}, alpha=0.6)
        plt.title(f'Spatial Distribution of ROIs - {interaction}')
        plt.savefig(plots_dir / 'spatial_distribution.png')
        plt.close()
        
        # 2. Score Distribution Boxplot
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=interaction_df, x='surrogate_label', y='score')
        plt.title(f'Score Distribution by Surrogate Label - {interaction}')
        plt.xlabel('Surrogate Label')
        plt.ylabel('Score')
        plt.tight_layout()
        plt.savefig(plots_dir / 'score_distribution.png')
        plt.close()
        
        # Step 3: Train and evaluate classifier
        # Prepare features and target
        X = interaction_df[['x', 'y', 'score']]
        y = interaction_df['surrogate_label']
        
        # For very small datasets (less than 10 samples), use a simple train-test split
        if len(X) < 10:
            logger.info(f"Using simple train-test split for {interaction} with {len(X)} samples")
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42, stratify=y)
            
            # Train classifier
            clf = LogisticRegression(random_state=42)
            clf.fit(X_train, y_train)
            
            # Evaluate
            accuracy = clf.score(X_test, y_test)
            
            # Save results
            results = {
                'interaction': interaction,
                'accuracy': accuracy,
                'n_samples': len(X),
                'class_distribution': class_counts.to_dict()
            }
        # For small datasets (10-50 samples), use k-fold cross validation
        elif len(X) <= 50:
            logger.info(f"Using k-fold cross validation for {interaction} with {len(X)} samples")
            n_splits = min(5, len(X) // 2)  # Use at most 5 folds
            clf = LogisticRegression(random_state=42)
            scores = cross_val_score(clf, X, y, cv=n_splits)
            accuracy = scores.mean()
            std = scores.std()
            
            logger.info(f"Cross-validation results for {interaction}:")
            logger.info(f"Mean accuracy: {accuracy:.3f} (+/- {std:.3f})")
            
            # Train final model on all data
            clf.fit(X, y)
            
            # Save results
            results = {
                'interaction': interaction,
                'accuracy': accuracy,
                'std': std,
                'n_samples': len(X),
                'class_distribution': class_counts.to_dict()
            }
        # For larger datasets, use train-test split
        else:
            logger.info(f"Using train-test split for {interaction} with {len(X)} samples")
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
            
            # Train classifier
            clf = LogisticRegression(random_state=42)
            clf.fit(X_train, y_train)
            
            # Evaluate
            accuracy = clf.score(X_test, y_test)
            
            # Save results
            results = {
                'interaction': interaction,
                'accuracy': accuracy,
                'n_samples': len(X),
                'class_distribution': class_counts.to_dict()
            }
        
        # Step 4: Analyze embeddings using t-SNE and UMAP
        analyze_embeddings(interaction_df, interaction, output_dir)
        
        logger.info(f"Completed analysis for {interaction}")
        logger.info(f"Results saved to {plots_dir}")

if __name__ == "__main__":
    surrogate_labeling() 