import torch
import numpy as np
import pandas as pd
from pathlib import Path
import logging
from torch_geometric.utils import dense_to_sparse
import networkx as nx
import matplotlib.pyplot as plt
import pickle

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_spatial_graph(positions, radius=30):
    """
    Create a spatial graph based on radius-based neighborhood
    
    Args:
        positions (torch.Tensor): Node positions (x, y, z)
        radius (float): Maximum distance for connecting nodes
        
    Returns:
        tuple: (edge_index, edge_weights)
    """
    # Calculate pairwise distances
    dist = torch.cdist(positions, positions)
    
    # Create adjacency matrix based on radius
    adj = (dist <= radius).float()
    
    # Remove self-loops
    adj.fill_diagonal_(0)
    
    # Calculate edge weights based on distance
    edge_weights = torch.exp(-dist / radius) * adj
    
    # Convert to edge index and weights
    edge_index = dense_to_sparse(adj)[0]
    edge_weights = edge_weights[edge_index[0], edge_index[1]]
    
    return edge_index, edge_weights

def calculate_node_weight(row, marker):
    """
    Calculate node weight based on marker expression and SpaFGAN score
    
    Args:
        row: DataFrame row containing marker values
        marker: Current marker being processed
    
    Returns:
        float: Node weight between 0 and 1
    """
    # Base weight from SpaFGAN score
    base_weight = float(row['spafgan_score'])
    
    # Additional weight from marker expression
    marker_weight = float(row[marker])
    
    # Combine weights (70% SpaFGAN, 30% marker expression)
    node_weight = 0.7 * base_weight + 0.3 * marker_weight
    
    return min(1.0, max(0.0, node_weight))

def calculate_link_weight(pos1, pos2, marker1, marker2, radius=30):
    """
    Calculate link weight based on spatial distance and marker compatibility
    
    Args:
        pos1, pos2: Node positions (x,y,z)
        marker1, marker2: Marker types
        radius: Maximum connection radius
    
    Returns:
        float: Link weight between 0 and 1
    """
    # Convert tensors to numpy arrays
    pos1 = pos1.cpu().numpy()
    pos2 = pos2.cpu().numpy()
    
    # Calculate spatial distance
    dist = np.sqrt(np.sum((pos1 - pos2) ** 2))
    
    if dist > radius:
        return 0.0
    
    # Base weight from distance (closer = higher weight)
    spatial_weight = np.exp(-dist / radius)
    
    # Marker compatibility weights
    compatibility_weights = {
        ("CD31", "CD11b"): 1.6,   # Strong: myeloid cells interacting with vasculature
        ("CD31", "CD11c"): 1.5,   # Strong: dendritic cells near endothelial layers
        ("CD31", "CD4"): 1.3,     # Moderate: T cell migration through vessels
        ("CD31", "CD20"): 1.2,    # Moderate: B cell circulation near vasculature
        ("CD31", "Catalase"): 1.0, # Weak: general oxidative stress near vessels
        
        ("CD11b", "CD11c"): 1.6,  # Strong: myeloid lineage coordination
        ("CD11b", "CD4"): 1.3,    # Moderate: T cell activation by myeloid cells
        ("CD11b", "CD20"): 1.2,   # Moderate: general immune co-localization
        ("CD11b", "Catalase"): 1.3, # Moderate: inflammation with oxidative stress
        
        ("CD11c", "CD4"): 1.5,    # Strong: dendritic priming T cells
        ("CD11c", "CD20"): 1.3,   # Moderate: APC–B cell neighborhood
        ("CD11c", "Catalase"): 1.3, # Moderate: oxidative microenvironments with APCs
        
        ("CD4", "CD20"): 1.2,     # Moderate: lymphoid zone interaction
    }
    
    # Get compatibility weight (default to 1.0 if not specified)
    marker_key = tuple(sorted([marker1, marker2]))
    compatibility_weight = compatibility_weights.get(marker_key, 1.0)
    
    # Combine weights (70% spatial, 30% compatibility)
    link_weight = 0.7 * spatial_weight + 0.3 * (compatibility_weight / 1.6)
    
    return min(1.0, max(0.0, link_weight))

def build_interaction_graph(roi_cells_path, output_dir, radius=30):
    """
    Build interaction graph from ROI cells
    
    Args:
        roi_cells_path (Path): Path to ROI cells CSV file
        output_dir (Path): Directory to save output files
        radius (float): Maximum distance for connecting nodes
    """
    try:
        logger.info(f"Building interaction graph from: {roi_cells_path}")
        
        # Load ROI cells
        df = pd.read_csv(roi_cells_path)
        if len(df) < 2:
            logger.warning(f"Not enough cells in {roi_cells_path} to build a graph (minimum 2 required)")
            return None
            
        logger.info(f"Loaded {len(df)} ROI cells")
        
        # Extract positions and features
        positions = torch.tensor(df[["x", "y", "z"]].values, dtype=torch.float32)
        
        # Create NetworkX graph
        G = nx.Graph()
        
        # Add nodes with calculated weights
        marker = roi_cells_path.stem.split('_')[-1]  # Extract marker from filename
        for idx, row in df.iterrows():
            try:
                node_weight = calculate_node_weight(row, marker)
                G.add_node(idx, 
                          pos=(row['x'], row['y']),
                          marker=marker,
                          spafgan_score=row['spafgan_score'],
                          weight=node_weight)
            except Exception as e:
                logger.error(f"Error adding node {idx}: {str(e)}")
                continue
        
        # Add edges with calculated weights
        edge_count = 0
        for i in range(len(df)):
            for j in range(i+1, len(df)):
                try:
                    pos1 = positions[i]
                    pos2 = positions[j]
                    link_weight = calculate_link_weight(pos1, pos2, marker, marker, radius)
                    if link_weight > 0:
                        G.add_edge(i, j, weight=link_weight)
                        edge_count += 1
                except Exception as e:
                    logger.error(f"Error adding edge between nodes {i} and {j}: {str(e)}")
                    continue
        
        if edge_count == 0:
            logger.warning(f"No edges were created for {marker}. Consider increasing the radius parameter.")
            return None
            
        logger.info(f"Created {edge_count} edges")
        
        # Save graph as pickle
        output_path = output_dir / f"interaction_graph_{roi_cells_path.stem}.pkl"
        with open(output_path, 'wb') as f:
            pickle.dump(G, f)
        logger.info(f"Saved graph to: {output_path}")
        
        # Visualize graph only if there are edges
        if G.number_of_edges() > 0:
            plt.figure(figsize=(12, 8))
            pos = nx.get_node_attributes(G, 'pos')
            node_weights = [G.nodes[n]['weight'] for n in G.nodes()]
            edge_weights = [G[u][v]['weight'] for u, v in G.edges()]
            
            # Create scatter plot for nodes
            scatter = plt.scatter([pos[n][0] for n in G.nodes()],
                                [pos[n][1] for n in G.nodes()],
                                c=node_weights,
                                s=100,
                                cmap=plt.cm.viridis)
            
            # Add edges
            for (u, v) in G.edges():
                plt.plot([pos[u][0], pos[v][0]],
                        [pos[u][1], pos[v][1]],
                        'k-',
                        alpha=G[u][v]['weight'],
                        linewidth=2)
            
            plt.colorbar(scatter, label='Node Weight')
            plt.title(f"Interaction Graph - {roi_cells_path.stem}")
            plt.axis('equal')
            
            # Save visualization
            vis_path = output_dir / f"interaction_graph_{roi_cells_path.stem}.png"
            plt.savefig(vis_path, dpi=300, bbox_inches='tight')
            plt.close()
            logger.info(f"Saved visualization to: {vis_path}")
        
        return G
        
    except Exception as e:
        logger.error(f"Error building interaction graph: {str(e)}")
        return None

def main():
    try:
        backend_dir = Path(__file__).parent.resolve()
        output_dir = backend_dir / "output"
        
        # Process each marker's ROI cells
        for marker in ["CD31", "CD11b", "CD11c"]:
            roi_path = output_dir / f"roi_cells_{marker}.csv"
            if roi_path.exists():
                logger.info(f"\nProcessing {marker}...")
                G = build_interaction_graph(roi_path, output_dir)
                
                if G is None:
                    logger.warning(f"Skipping statistics for {marker} due to graph creation failure")
                    continue
                
                # Print detailed graph statistics
                logger.info(f"\n📊 Detailed Graph Statistics for {marker}:")
                logger.info("=" * 50)
                
                try:
                    # Basic metrics
                    logger.info("\nBasic Metrics:")
                    logger.info(f"Number of nodes: {G.number_of_nodes()}")
                    logger.info(f"Number of edges: {G.number_of_edges()}")
                    
                    if G.number_of_nodes() > 0:
                        logger.info(f"Average degree: {sum(dict(G.degree()).values()) / G.number_of_nodes():.2f}")
                        logger.info(f"Graph density: {nx.density(G):.4f}")
                    
                    # Centrality metrics
                    if G.number_of_nodes() > 1:
                        logger.info("\nCentrality Metrics:")
                        try:
                            degree_centrality = nx.degree_centrality(G)
                            logger.info(f"Average degree centrality: {sum(degree_centrality.values()) / len(G):.4f}")
                        except Exception as e:
                            logger.error(f"Error calculating degree centrality: {str(e)}")
                        
                        try:
                            betweenness_centrality = nx.betweenness_centrality(G)
                            logger.info(f"Average betweenness centrality: {sum(betweenness_centrality.values()) / len(G):.4f}")
                        except Exception as e:
                            logger.error(f"Error calculating betweenness centrality: {str(e)}")
                        
                        try:
                            closeness_centrality = nx.closeness_centrality(G)
                            logger.info(f"Average closeness centrality: {sum(closeness_centrality.values()) / len(G):.4f}")
                        except Exception as e:
                            logger.error(f"Error calculating closeness centrality: {str(e)}")
                    
                    # Component analysis
                    logger.info("\nComponent Analysis:")
                    components = list(nx.connected_components(G))
                    logger.info(f"Number of connected components: {len(components)}")
                    if components:
                        logger.info(f"Largest component size: {len(max(components, key=len))}")
                        logger.info(f"Average component size: {sum(len(c) for c in components) / len(components):.2f}")
                    
                    # Clustering metrics
                    if G.number_of_nodes() > 1:
                        logger.info("\nClustering Metrics:")
                        try:
                            logger.info(f"Average clustering coefficient: {nx.average_clustering(G):.4f}")
                        except Exception as e:
                            logger.error(f"Error calculating clustering coefficient: {str(e)}")
                        
                        try:
                            logger.info(f"Transitivity: {nx.transitivity(G):.4f}")
                        except Exception as e:
                            logger.error(f"Error calculating transitivity: {str(e)}")
                    
                    # Path metrics
                    if G.number_of_nodes() > 1:
                        logger.info("\nPath Metrics:")
                        largest_cc = max(components, key=len)
                        if len(largest_cc) > 1:
                            largest_cc_graph = G.subgraph(largest_cc)
                            try:
                                logger.info(f"Average shortest path length: {nx.average_shortest_path_length(largest_cc_graph):.2f}")
                                logger.info(f"Diameter: {nx.diameter(largest_cc_graph)}")
                            except Exception as e:
                                logger.error(f"Error calculating path metrics: {str(e)}")
                    
                    # SpaFGAN score statistics
                    if G.number_of_nodes() > 0:
                        logger.info("\nSpaFGAN Score Statistics:")
                        spafgan_scores = [G.nodes[n]['spafgan_score'] for n in G.nodes()]
                        logger.info(f"Average SpaFGAN score: {np.mean(spafgan_scores):.4f}")
                        logger.info(f"Max SpaFGAN score: {np.max(spafgan_scores):.4f}")
                        logger.info(f"Min SpaFGAN score: {np.min(spafgan_scores):.4f}")
                
                except Exception as e:
                    logger.error(f"Error calculating statistics for {marker}: {str(e)}")
                    continue
                
            else:
                logger.warning(f"ROI cells file not found for {marker}: {roi_path}")
    
    except Exception as e:
        logger.error(f"Error in main function: {str(e)}")
        raise

if __name__ == "__main__":
    main()
