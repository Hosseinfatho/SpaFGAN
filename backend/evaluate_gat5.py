import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
import numpy as np
from train_gat4 import MarkerGAT, MARKERS
import os
from tabulate import tabulate
import logging
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, precision_score, recall_score, f1_score
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def evaluate_model(model, loader, device, model_type='GAT'):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    all_reconstructed = []
    all_original = []
    
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            
            if model_type == 'CD11cMLP':
                # For CD11c MLP model (no edges)
                out = model(batch.x)
            else:
                # For GAT models
                out = model(batch.x, batch.edge_index, batch.edge_attr)
            
            # Store original and reconstructed values for regression metrics
            all_original.extend(batch.x.cpu().numpy())
            all_reconstructed.extend(out.cpu().numpy())
            
            loss = F.mse_loss(out, batch.x)
            total_loss += loss.item()
            
            # Convert predictions to binary (high/low expression)
            # Use global mean instead of batch mean
            preds = (out > 0.5).float()  # Use fixed threshold
            labels = (batch.x > 0.5).float()  # Use fixed threshold
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate regression metrics on original values
    all_original = np.array(all_original)
    all_reconstructed = np.array(all_reconstructed)
    
    # Flatten arrays for metrics calculation
    original_flat = all_original.flatten()
    reconstructed_flat = all_reconstructed.flatten()
    
    # Remove any NaN or Inf values
    valid_mask = ~(np.isnan(original_flat) | np.isinf(original_flat) | 
                   np.isnan(reconstructed_flat) | np.isinf(reconstructed_flat))
    
    if np.sum(valid_mask) == 0:
        logger.warning("No valid values for metrics calculation")
        return {
            'loss': float('inf'),
            'mse': float('inf'),
            'rmse': float('inf'),
            'mae': float('inf'),
            'r2': 0.0,
            'accuracy': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0
        }
    
    original_flat = original_flat[valid_mask]
    reconstructed_flat = reconstructed_flat[valid_mask]
    
    # Calculate regression metrics
    mse = mean_squared_error(original_flat, reconstructed_flat)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(original_flat, reconstructed_flat)
    
    # Calculate R² score
    try:
        r2 = r2_score(original_flat, reconstructed_flat)
    except:
        r2 = 0.0
    
    # Calculate classification metrics
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # Remove any NaN or Inf values from classification data
    valid_class_mask = ~(np.isnan(all_preds) | np.isinf(all_preds) | 
                        np.isnan(all_labels) | np.isinf(all_labels))
    
    if np.sum(valid_class_mask) == 0:
        logger.warning("No valid values for classification metrics")
        accuracy = 0.0
        precision = 0.0
        recall = 0.0
        f1 = 0.0
    else:
        all_preds = all_preds[valid_class_mask]
        all_labels = all_labels[valid_class_mask]
        
        accuracy = accuracy_score(all_labels.flatten(), all_preds.flatten())
        precision = precision_score(all_labels.flatten(), all_preds.flatten(), average='weighted', zero_division=0)
        recall = recall_score(all_labels.flatten(), all_preds.flatten(), average='weighted', zero_division=0)
        f1 = f1_score(all_labels.flatten(), all_preds.flatten(), average='weighted', zero_division=0)
    
    # Log some debugging information
    logger.info(f"Debug - Original range: [{original_flat.min():.3f}, {original_flat.max():.3f}]")
    logger.info(f"Debug - Reconstructed range: [{reconstructed_flat.min():.3f}, {reconstructed_flat.max():.3f}]")
    logger.info(f"Debug - MSE: {mse:.6f}, RMSE: {rmse:.6f}, MAE: {mae:.6f}, R²: {r2:.6f}")
    
    return {
        'loss': total_loss / len(loader) if len(loader) > 0 else float('inf'),
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

def print_marker_results(marker, train_metrics, test_metrics):
    logger.info(f"\n{'='*50}")
    logger.info(f"Detailed Results for {marker}")
    logger.info(f"{'='*50}")
    
    logger.info("\nTraining Metrics:")
    logger.info(f"Loss: {train_metrics['loss']:.6f}")
    logger.info(f"MSE: {train_metrics['mse']:.6f}")
    logger.info(f"RMSE: {train_metrics['rmse']:.6f}")
    logger.info(f"MAE: {train_metrics['mae']:.6f}")
    logger.info(f"R² Score: {train_metrics['r2']:.6f}")
    logger.info(f"Accuracy: {train_metrics['accuracy']:.6f}")
    logger.info(f"Precision: {train_metrics['precision']:.6f}")
    logger.info(f"Recall: {train_metrics['recall']:.6f}")
    logger.info(f"F1 Score: {train_metrics['f1']:.6f}")
    
    logger.info("\nTest Metrics:")
    logger.info(f"Loss: {test_metrics['loss']:.6f}")
    logger.info(f"MSE: {test_metrics['mse']:.6f}")
    logger.info(f"RMSE: {test_metrics['rmse']:.6f}")
    logger.info(f"MAE: {test_metrics['mae']:.6f}")
    logger.info(f"R² Score: {test_metrics['r2']:.6f}")
    logger.info(f"Accuracy: {test_metrics['accuracy']:.6f}")
    logger.info(f"Precision: {test_metrics['precision']:.6f}")
    logger.info(f"Recall: {test_metrics['recall']:.6f}")
    logger.info(f"F1 Score: {test_metrics['f1']:.6f}")
    
    logger.info(f"\nPerformance Summary:")
    logger.info(f"R² Score Difference (Train-Test): {train_metrics['r2'] - test_metrics['r2']:.6f}")
    logger.info(f"RMSE Difference (Train-Test): {train_metrics['rmse'] - test_metrics['rmse']:.6f}")
    logger.info(f"Accuracy Difference (Train-Test): {train_metrics['accuracy'] - test_metrics['accuracy']:.6f}")
    logger.info(f"F1 Score Difference (Train-Test): {train_metrics['f1'] - test_metrics['f1']:.6f}")

def extract_subgraphs_from_main(main_graph):
    """Extract subgraphs from main graph for evaluation"""
    from torch_geometric.data import Data
    
    subgraphs = []
    metadata = main_graph.metadata
    
    for subgraph_info in metadata['subgraphs']:
        start_node = subgraph_info['start_node']
        end_node = subgraph_info['end_node']
        
        # Extract node features for this subgraph
        subgraph_x = main_graph.x[start_node:end_node + 1]
        
        # Extract edges that are within this subgraph
        edge_mask = ((main_graph.edge_index[0] >= start_node) & 
                    (main_graph.edge_index[0] <= end_node) &
                    (main_graph.edge_index[1] >= start_node) & 
                    (main_graph.edge_index[1] <= end_node))
        
        subgraph_edge_index = main_graph.edge_index[:, edge_mask]
        subgraph_edge_attr = main_graph.edge_attr[edge_mask]
        
        # Adjust edge indices to be local to this subgraph
        subgraph_edge_index = subgraph_edge_index - start_node
        
        # Create subgraph Data object
        subgraph = Data(
            x=subgraph_x,
            edge_index=subgraph_edge_index,
            edge_attr=subgraph_edge_attr,
            roi_id=subgraph_info['roi_id'],
            center=subgraph_info['center']
        )
        
        subgraphs.append(subgraph)
    
    return subgraphs

def main():
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    markers = MARKERS
    table_data = []
    results = []
    
    logger.info("\nStarting evaluation...")
    
    for marker in markers:
        logger.info(f"\nEvaluating model for {marker}")
        
        # Load main graph
        main_graph_path = f'output/main_marker_graphs/{marker}_main_graph.pt'
        if not os.path.exists(main_graph_path):
            logger.error(f"Main graph file not found: {main_graph_path}")
            continue
            
        main_graph = torch.load(main_graph_path)
        
        # Extract subgraphs
        subgraphs = extract_subgraphs_from_main(main_graph)
        
        if len(subgraphs) < 2:
            logger.warning(f"Not enough subgraphs for {marker}: {len(subgraphs)}")
            continue
        
        # Split data into train and test sets
        train_size = int(0.85 * len(subgraphs))
        train_data = subgraphs[:train_size]
        test_data = subgraphs[train_size:]
        
        train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_data, batch_size=32, shuffle=False)
        
        # Load model
        checkpoint_path = f'output/best_model_{marker}.pt'
        if not os.path.exists(checkpoint_path):
            logger.error(f"Model checkpoint not found: {checkpoint_path}")
            continue
            
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Determine model type
        model_type = checkpoint.get('model_type', 'GAT')
        
        if model_type == 'CD11cMLP':
            # Load CD11c MLP model
            from train_gat4 import CD11cMLP
            model = CD11cMLP(in_channels=len(MARKERS))
        else:
            # Load GAT model
            model = MarkerGAT(in_channels=len(MARKERS), hidden_channels=64, out_channels=32)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        
        # Evaluate model
        train_metrics = evaluate_model(model, train_loader, device, model_type)
        test_metrics = evaluate_model(model, test_loader, device, model_type)
        
        print_marker_results(marker, train_metrics, test_metrics)
        
        # Add results to table data
        table_data.append([
            marker,
            len(train_data),
            len(test_data),
            f"{train_metrics['loss']:.4f}",
            f"{test_metrics['loss']:.4f}",
            f"{train_metrics['rmse']:.4f}",
            f"{test_metrics['rmse']:.4f}",
            f"{train_metrics['mae']:.4f}",
            f"{test_metrics['mae']:.4f}",
            f"{train_metrics['r2']:.4f}",
            f"{test_metrics['r2']:.4f}",
            f"{train_metrics['accuracy']:.4f}",
            f"{test_metrics['accuracy']:.4f}",
            f"{train_metrics['recall']:.4f}",
            f"{test_metrics['recall']:.4f}",
            f"{train_metrics['f1']:.4f}",
            f"{test_metrics['f1']:.4f}"
        ])
        
        # Add results to JSON data
        results.append({
            'marker': marker,
            'model_type': model_type,
            'train_size': len(train_data),
            'test_size': len(test_data),
            'train_metrics': {
                'loss': float(train_metrics['loss']),
                'mse': float(train_metrics['mse']),
                'rmse': float(train_metrics['rmse']),
                'mae': float(train_metrics['mae']),
                'r2': float(train_metrics['r2']),
                'accuracy': float(train_metrics['accuracy']),
                'precision': float(train_metrics['precision']),
                'recall': float(train_metrics['recall']),
                'f1': float(train_metrics['f1'])
            },
            'test_metrics': {
                'loss': float(test_metrics['loss']),
                'mse': float(test_metrics['mse']),
                'rmse': float(test_metrics['rmse']),
                'mae': float(test_metrics['mae']),
                'r2': float(test_metrics['r2']),
                'accuracy': float(test_metrics['accuracy']),
                'precision': float(test_metrics['precision']),
                'recall': float(test_metrics['recall']),
                'f1': float(test_metrics['f1'])
            }
        })
    
    logger.info(f"\n{'='*50}")
    logger.info("Evaluation Summary")
    logger.info(f"{'='*50}")
    
    # Display results table
    headers = ['Marker', 'Train', 'Test', 
              'TrLoss', 'TeLoss', 
              'TrRMSE', 'TeRMSE', 
              'TrMAE', 'TeMAE', 
              'TrR2', 'TeR2',
              'TrAcc', 'TeAcc',
              'TrRec', 'TeRec',
              'TrF1', 'TeF1']
    
    # Format table data with shorter numbers
    formatted_data = []
    for row in table_data:
        formatted_row = [
            row[0],  # Marker name
            row[1],  # Train size
            row[2],  # Test size
            f"{float(row[3]):.3f}",  # Train Loss
            f"{float(row[4]):.3f}",  # Test Loss
            f"{float(row[5]):.3f}",  # Train RMSE
            f"{float(row[6]):.3f}",  # Test RMSE
            f"{float(row[7]):.3f}",  # Train MAE
            f"{float(row[8]):.3f}",  # Test MAE
            f"{float(row[9]):.3f}",  # Train R2
            f"{float(row[10]):.3f}", # Test R2
            f"{float(row[11]):.3f}", # Train Acc
            f"{float(row[12]):.3f}", # Test Acc
            f"{float(row[13]):.3f}", # Train Recall
            f"{float(row[14]):.3f}", # Test Recall
            f"{float(row[15]):.3f}", # Train F1
            f"{float(row[16]):.3f}"  # Test F1
        ]
        formatted_data.append(formatted_row)
    
    print("\nResults Table:")
    print(tabulate(formatted_data, headers=headers, tablefmt='simple', numalign='right', stralign='left'))
    
    # Save results to JSON
    os.makedirs('output', exist_ok=True)
    results_file = 'output/gat_evaluation_results.json'
    
    # Save results
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=4)
    
    logger.info(f"\nResults have been saved to {results_file}")

if __name__ == "__main__":
    main() 