import torch
import numpy as np
import logging
from pathlib import Path
from train_gat4 import MarkerGAT, INTERACTIONS, MARKERS
from Extract_ROI6 import compute_interaction_score_with_model, compute_interaction_score_fallback
from torch_geometric.data import Data
from scipy.stats import spearmanr, pearsonr

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_multiple_test_graphs(num_graphs=20):
    """Create multiple test graphs with varying quality"""
    graphs = []
    
    for i in range(num_graphs):
        # Create graphs with different quality levels
        num_nodes = np.random.randint(5, 15)
        
        # Vary the quality of marker expressions
        quality_factor = np.random.uniform(0.3, 1.0)
        
        x = torch.rand(num_nodes, len(MARKERS)) * quality_factor
        edge_index = torch.randint(0, num_nodes, (2, num_nodes * 2))
        edge_attr = torch.rand(edge_index.size(1), 1)
        pos = torch.rand(num_nodes, 2)
        
        graph = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            pos=pos
        )
        
        graphs.append(graph)
    
    return graphs

def test_ranking_correlation():
    """Test if the improved method preserves relative rankings"""
    logger.info("=" * 60)
    logger.info("TESTING RANKING CORRELATION")
    logger.info("=" * 60)
    
    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_paths = ["output/best_model_CD31.pt", "output/best_model_CD20.pt", "output/best_model_CD11b.pt"]
    
    model = None
    for model_path in model_paths:
        if Path(model_path).exists():
            try:
                model = MarkerGAT(in_channels=len(MARKERS), hidden_channels=64, out_channels=32).to(device)
                checkpoint = torch.load(model_path, map_location=device)
                model.load_state_dict(checkpoint['model_state_dict'])
                logger.info(f"Loaded model from {model_path}")
                break
            except Exception as e:
                logger.error(f"Error loading {model_path}: {str(e)}")
    
    if model is None:
        logger.error("No model could be loaded!")
        return
    
    # Create test graphs
    test_graphs = create_multiple_test_graphs(50)
    
    # Test each interaction
    for interaction_name in INTERACTIONS.keys():
        logger.info(f"\nTesting ranking correlation for: {interaction_name}")
        logger.info("-" * 50)
        
        original_scores = []
        improved_scores = []
        
        for i, graph in enumerate(test_graphs):
            try:
                graph = graph.to(device)
                
                # Original method
                original_score = compute_interaction_score_fallback(graph, interaction_name)
                original_scores.append(original_score)
                
                # Improved method
                improved_score = compute_interaction_score_with_model(model, graph, interaction_name)
                improved_scores.append(improved_score)
                
            except Exception as e:
                logger.warning(f"Error processing graph {i}: {str(e)}")
                continue
        
        if len(original_scores) > 5:  # Need enough data for correlation
            # Calculate correlations
            spearman_corr, spearman_p = spearmanr(original_scores, improved_scores)
            pearson_corr, pearson_p = pearsonr(original_scores, improved_scores)
            
            logger.info(f"Number of valid graphs: {len(original_scores)}")
            logger.info(f"Spearman correlation: {spearman_corr:.4f} (p={spearman_p:.4f})")
            logger.info(f"Pearson correlation: {pearson_corr:.4f} (p={pearson_p:.4f})")
            
            # Check if correlation is significant
            if spearman_p < 0.05:
                logger.info("✅ Significant correlation - rankings are preserved!")
            else:
                logger.info("⚠️  No significant correlation - rankings may differ")
            
            # Show top 5 rankings
            original_ranked = sorted(enumerate(original_scores), key=lambda x: x[1], reverse=True)
            improved_ranked = sorted(enumerate(improved_scores), key=lambda x: x[1], reverse=True)
            
            logger.info("\nTop 5 rankings comparison:")
            logger.info("Original | Improved")
            logger.info("---------|---------")
            for i in range(min(5, len(original_scores))):
                orig_idx, orig_score = original_ranked[i]
                imp_idx, imp_score = improved_ranked[i]
                logger.info(f"Graph {orig_idx:2d} ({orig_score:.3f}) | Graph {imp_idx:2d} ({imp_score:.3f})")
        else:
            logger.warning("Not enough valid graphs for correlation analysis")

def test_score_distributions():
    """Test the distribution of scores from both methods"""
    logger.info("\n" + "=" * 60)
    logger.info("TESTING SCORE DISTRIBUTIONS")
    logger.info("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_paths = ["output/best_model_CD31.pt", "output/best_model_CD20.pt", "output/best_model_CD11b.pt"]
    
    model = None
    for model_path in model_paths:
        if Path(model_path).exists():
            try:
                model = MarkerGAT(in_channels=len(MARKERS), hidden_channels=64, out_channels=32).to(device)
                checkpoint = torch.load(model_path, map_location=device)
                model.load_state_dict(checkpoint['model_state_dict'])
                break
            except Exception as e:
                continue
    
    if model is None:
        logger.error("No model could be loaded!")
        return
    
    test_graphs = create_multiple_test_graphs(100)
    
    for interaction_name in INTERACTIONS.keys():
        logger.info(f"\nScore distributions for: {interaction_name}")
        logger.info("-" * 40)
        
        original_scores = []
        improved_scores = []
        
        for graph in test_graphs:
            try:
                graph = graph.to(device)
                original_score = compute_interaction_score_fallback(graph, interaction_name)
                improved_score = compute_interaction_score_with_model(model, graph, interaction_name)
                
                original_scores.append(original_score)
                improved_scores.append(improved_score)
            except:
                continue
        
        if original_scores and improved_scores:
            orig_mean = np.mean(original_scores)
            orig_std = np.std(original_scores)
            imp_mean = np.mean(improved_scores)
            imp_std = np.std(improved_scores)
            
            logger.info(f"Original: Mean={orig_mean:.4f}, Std={orig_std:.4f}")
            logger.info(f"Improved: Mean={imp_mean:.4f}, Std={imp_std:.4f}")
            logger.info(f"Score range - Original: [{min(original_scores):.4f}, {max(original_scores):.4f}]")
            logger.info(f"Score range - Improved: [{min(improved_scores):.4f}, {max(improved_scores):.4f}]")

def main():
    """Main test function"""
    logger.info("Starting ranking correlation analysis...")
    
    # Test ranking correlations
    test_ranking_correlation()
    
    # Test score distributions
    test_score_distributions()
    
    logger.info("\n" + "=" * 60)
    logger.info("RANKING ANALYSIS COMPLETED")
    logger.info("=" * 60)

if __name__ == "__main__":
    main() 