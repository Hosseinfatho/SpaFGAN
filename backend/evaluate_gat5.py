import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
import numpy as np
from train_gat4 import MarkerGAT
import os
from tabulate import tabulate
import logging
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, precision_score, recall_score, f1_score
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def evaluate_model(model, loader, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index, batch.edge_attr)
            loss = F.mse_loss(out, batch.x)
            total_loss += loss.item()
            
            # Convert predictions to binary (high/low expression)
            preds = (out > batch.x.mean(dim=0)).float()
            labels = (batch.x > batch.x.mean(dim=0)).float()
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate evaluation metrics
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    mse = mean_squared_error(all_labels.flatten(), all_preds.flatten())
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(all_labels.flatten(), all_preds.flatten())
    r2 = r2_score(all_labels.flatten(), all_preds.flatten())
    
    # Calculate classification metrics
    accuracy = accuracy_score(all_labels.flatten(), all_preds.flatten())
    precision = precision_score(all_labels.flatten(), all_preds.flatten(), average='weighted')
    recall = recall_score(all_labels.flatten(), all_preds.flatten(), average='weighted')
    f1 = f1_score(all_labels.flatten(), all_preds.flatten(), average='weighted')
    
    return {
        'loss': total_loss / len(loader),
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

def main():
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    markers = ['CD31', 'CD20', 'CD11b', 'CD4', 'CD11c', 'Catalase']
    table_data = []
    results = []
    
    logger.info("\nStarting evaluation...")
    
    for marker in markers:
        logger.info(f"\nEvaluating GAT model for {marker}")
        
        # Load data
        data_dir = f'output/roi_graphs_{marker}.pt'
        if not os.path.exists(data_dir):
            logger.error(f"Data file not found: {data_dir}")
            continue
            
        data = torch.load(data_dir)
        graphs = []
        for graph_dict in data['graphs']:
            graph = graph_dict['graph']
            if isinstance(graph, dict):
                from torch_geometric.data import Data
                graph = Data(
                    x=graph['x'],
                    edge_index=graph['edge_index'],
                    edge_attr=graph['edge_attr'],
                    pos=graph['pos'],
                    marker=graph['marker']
                )
            graphs.append(graph)
        
        # Split data into train and test sets
        train_size = int(0.85 * len(graphs))
        train_data = graphs[:train_size]
        test_data = graphs[train_size:]
        
        train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_data, batch_size=32, shuffle=False)
        
        # Load model
        model = MarkerGAT(in_channels=6, hidden_channels=64, out_channels=32)
        checkpoint = torch.load(f'output/best_model_{marker}.pt')
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        
        # Evaluate model
        train_metrics = evaluate_model(model, train_loader, device)
        test_metrics = evaluate_model(model, test_loader, device)
        
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