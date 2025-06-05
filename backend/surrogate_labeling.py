import pandas as pd
import numpy as np
from pathlib import Path
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
from umap import UMAP
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def analyze_embeddings(df, marker, output_dir):
    """
    Analyze ROI embeddings using t-SNE and UMAP visualization.
    
    Args:
        df (pd.DataFrame): DataFrame containing ROI data
        marker (str): Current marker being analyzed
        output_dir (Path): Directory to save output files
    """
    # Create embeddings directory
    embeddings_dir = output_dir / "embeddings" / marker
    embeddings_dir.mkdir(parents=True, exist_ok=True)
    
    # Prepare features for embedding
    marker_columns = ['CD31', 'CD20', 'CD11b', 'CD4', 'CD11c', 'Catalase']
    X = df[marker_columns].values
    X_scaled = StandardScaler().fit_transform(X)
    # Apply t-SNE
    tsne = TSNE(n_components=2, perplexity=5, random_state=42)
    tsne_embeddings = tsne.fit_transform(X_scaled)
    
    # Apply UMAP
    umap = UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
    umap_embeddings = umap.fit_transform(X_scaled)
    
    # Create visualizations
    for method, embeddings in [("t-SNE", tsne_embeddings), ("UMAP", umap_embeddings)]:
        plt.figure(figsize=(10, 10))
        df_vis = pd.DataFrame(embeddings, columns=["Dim1", "Dim2"])
        df_vis["Label"] = df['surrogate_label']
        
        sns.scatterplot(data=df_vis, x="Dim1", y="Dim2", hue="Label", 
                       palette={0: 'blue', 1: 'red'}, alpha=0.6)
        plt.title(f"{method} of ROI Embeddings by Surrogate Label - {marker}")
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
    
    # List of markers to process
    markers = ["CD31", "CD11b", "CD11c"]
    
    for marker in markers:
        # Read the combined ROI file
        roi_path = output_dir / f"roi_cells_{marker}_all.csv"
        if not roi_path.exists():
            logger.warning(f"File not found: {roi_path}")
            continue
            
        df = pd.read_csv(roi_path)
        
        # Step 1: Assign surrogate labels
        # Group by interaction_type and assign labels
        def assign_labels(scores):
            # Sort scores
            sorted_scores = scores.sort_values(ascending=False)
            n = len(sorted_scores)
            labels = np.full(n, -1, dtype=int)  #
            top_n = int(0.1 * n)
            bottom_n = int(0.1 * n)
            if top_n == 0: top_n = 1
            if bottom_n == 0: bottom_n = 1
            labels[:top_n] = 1
            labels[-bottom_n:] = 0
            return pd.Series(labels, index=sorted_scores.index)
        
        df['surrogate_label'] = df.groupby('interaction_type')['interaction_score'].transform(assign_labels)

        df = df[df['surrogate_label'] != -1].reset_index(drop=True)

        # Step 2: Create visualizations
        # Create output directory for plots
        plots_dir = output_dir / "surrogate_plots" / marker
        plots_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. XY Spatial Scatter Plot
        plt.figure(figsize=(10, 10))
        sns.scatterplot(data=df, x='x', y='y', hue='surrogate_label', 
                       palette={0: 'blue', 1: 'red'}, alpha=0.6)
        plt.title(f'Spatial Distribution of ROIs - {marker}')
        plt.savefig(plots_dir / 'spatial_distribution.png')
        plt.close()
        
        # 2. Marker Expression Boxplots
        marker_columns = ['CD31', 'CD20', 'CD11b', 'CD4', 'CD11c', 'Catalase']
        plt.figure(figsize=(15, 10))
        df_melted = df.melt(id_vars=['surrogate_label'], 
                           value_vars=marker_columns,
                           var_name='Marker', value_name='Expression')
        sns.boxplot(data=df_melted, x='Marker', y='Expression', hue='surrogate_label')
        plt.title(f'Marker Expression by Surrogate Label - {marker}')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(plots_dir / 'marker_expression.png')
        plt.close()
        
        # Step 3: Train and evaluate classifier
        # Prepare features and target
        X = df[marker_columns]
        y = df['surrogate_label']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train classifier
        clf = LogisticRegression(random_state=42)
        clf.fit(X_train, y_train)
        
        # Make predictions
        y_pred = clf.predict(X_test)
        y_pred_proba = clf.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        auroc = roc_auc_score(y_test, y_pred_proba)
        
        # Save results
        results = {
            'marker': marker,
            'accuracy': accuracy,
            'auroc': auroc,
            'feature_importance': dict(zip(marker_columns, clf.coef_[0]))
        }
        
        # Save results to file
        results_path = plots_dir / 'classification_results.txt'
        with open(results_path, 'w') as f:
            f.write(f"Results for {marker}\n")
            f.write("="*50 + "\n")
            f.write(f"Accuracy: {accuracy:.3f}\n")
            f.write(f"AUROC: {auroc:.3f}\n")
            f.write("\nFeature Importance:\n")
            for feature, importance in results['feature_importance'].items():
                f.write(f"{feature}: {importance:.3f}\n")
        
        # Step 4: Analyze embeddings using t-SNE and UMAP
        analyze_embeddings(df, marker, output_dir)
        
        logger.info(f"Completed analysis for {marker}")
        logger.info(f"Results saved to {plots_dir}")

if __name__ == "__main__":
    surrogate_labeling() 