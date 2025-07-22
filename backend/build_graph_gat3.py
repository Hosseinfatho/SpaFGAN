import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.data import Data
import logging
from pathlib import Path
from tqdm import tqdm
import os

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define output directory
OUTPUT_DIR = "output"

# Define markers
MARKERS = ["CD31", "CD20", "CD11b", "CD4", "CD11c", "Catalase"]

# Define biological interactions - updated interaction names and pairs
INTERACTIONS = {
    "B-cell infiltration": ["CD20", "CD31"],      # CD20⁺ CD31⁺
    "T-cell maturation": ["CD4", "CD31"],         # CD4⁺ CD31⁺
    "Inflammatory zone": ["CD11b", "CD11c"],      # CD11b⁺ CD11c⁺
    "Oxidative stress regulation": ["CD11b", "Catalase"]  # CD11b⁺ Catalase⁺
}

# Define meaningful marker pairs for inter-connections - updated for new interactions
MEANINGFUL_PAIRS = [
    ("CD20", "CD31"),  # B-cell infiltration
    ("CD4", "CD31"),   # T-cell maturation
    ("CD11b", "CD11c"), # Inflammatory zone
    ("CD11b", "Catalase"), # Oxidative stress regulation
]

# Define base radius range
MIN_RADIUS = 25.0
MAX_RADIUS = 40.0

class MarkerGAT(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=4):
        super(MarkerGAT, self).__init__()
        self.gat1 = GATConv(in_channels, hidden_channels, heads=heads)
        self.gat2 = GATConv(hidden_channels * heads, out_channels, heads=1)
        
    def forward(self, x, edge_index, edge_attr):
        # First GAT layer with edge attributes
        x = F.elu(self.gat1(x, edge_index, edge_attr))
        # Second GAT layer
        x = F.elu(self.gat2(x, edge_index))
        return x

def calculate_adaptive_radius(roi_row, all_cells_df, marker):
    """Calculate adaptive radius based on biological interactions and local density"""
    
    # Base radius
    base_radius = 25.0
    
    # Factor 1: Biological interaction strength
    interaction_factor = 1.0
    
    # Check if this marker is part of known interactions
    for interaction_name, markers in INTERACTIONS.items():
        if marker in markers:
            # Find other markers in this interaction
            other_markers = [m for m in markers if m != marker]
            
            # Calculate interaction strength based on presence of other markers
            interaction_strength = 0.0
            for other_marker in other_markers:
                if other_marker in all_cells_df.columns:
                    # Check if there are cells with high expression of other marker nearby
                    nearby_cells = all_cells_df[
                        (all_cells_df['x'] >= roi_row['x'] - 40) & 
                        (all_cells_df['x'] <= roi_row['x'] + 40) & 
                        (all_cells_df['y'] >= roi_row['y'] - 40) & 
                        (all_cells_df['y'] <= roi_row['y'] + 40)
                    ]
                    
                    if len(nearby_cells) > 0:
                        # Calculate average expression of other marker
                        avg_expr = nearby_cells[other_marker].mean()
                        interaction_strength += avg_expr
            
            # Adjust interaction factor based on strength
            if interaction_strength > 0.5:  # High interaction
                interaction_factor = 1.3
            elif interaction_strength > 0.2:  # Medium interaction
                interaction_factor = 1.1
            else:  # Low interaction
                interaction_factor = 0.9
            break
    
    # Factor 2: Local cell density
    density_factor = 1.0
    
    # Count cells in base radius
    base_cells = all_cells_df[
        (all_cells_df['x'] >= roi_row['x'] - base_radius) & 
        (all_cells_df['x'] <= roi_row['x'] + base_radius) & 
        (all_cells_df['y'] >= roi_row['y'] - base_radius) & 
        (all_cells_df['y'] <= roi_row['y'] + base_radius)
    ]
    
    cell_count = len(base_cells)
    
    # Adjust based on density - HIGHER density = LARGER radius
    if cell_count > 20:  # High density - increase radius to find bigger ROI
        density_factor = 1.4
    elif cell_count > 10:  # Medium-high density - moderate increase
        density_factor = 1.2
    elif cell_count < 5:  # Low density - decrease radius
        density_factor = 0.8
    else:  # Medium density
        density_factor = 1.0
    
    # Factor 3: Marker-specific adjustments
    marker_factor = 1.0
    
    if marker == "CD31":  # Blood vessels - need larger radius
        marker_factor = 1.2
    elif marker == "CD20":  # B cells - medium radius
        marker_factor = 1.0
    elif marker == "CD11b":  # Immune cells - smaller radius
        marker_factor = 0.9
    elif marker == "CD4":  # T cells - medium radius
        marker_factor = 1.0
    elif marker == "CD11c":  # Dendritic cells - smaller radius
        marker_factor = 0.9
    elif marker == "Catalase":  # Oxidative stress - medium radius
        marker_factor = 1.0
    
    # Calculate final adaptive radius
    adaptive_radius = base_radius * interaction_factor * density_factor * marker_factor
    
    # Clamp to range [MIN_RADIUS, MAX_RADIUS]
    adaptive_radius = max(MIN_RADIUS, min(MAX_RADIUS, adaptive_radius))
    
    return adaptive_radius, {
        'base_radius': base_radius,
        'interaction_factor': interaction_factor,
        'density_factor': density_factor,
        'marker_factor': marker_factor,
        'cell_count': cell_count,
        'final_radius': adaptive_radius
    }

def find_neighbors_in_radius(roi_row, all_cells_df, marker):
    """Find all cells within adaptive radius of the ROI cell"""
    x, y = roi_row['x'], roi_row['y']
    
    # Calculate adaptive radius
    radius, radius_info = calculate_adaptive_radius(roi_row, all_cells_df, marker)
    
    # Find cells within radius
    mask = (
        (all_cells_df['x'] >= x - radius) & 
        (all_cells_df['x'] <= x + radius) & 
        (all_cells_df['y'] >= y - radius) & 
        (all_cells_df['y'] <= y + radius)
    )
    
    # Calculate actual distances
    neighbors = all_cells_df[mask].copy()
    if len(neighbors) > 0:
        neighbors['distance'] = np.sqrt(
            (neighbors['x'] - x)**2 + 
            (neighbors['y'] - y)**2
        )
        # Filter by actual distance
        neighbors = neighbors[neighbors['distance'] <= radius]
    
    return neighbors, radius_info

def build_roi_graph(roi_row, neighbors_df):
    """Build graph for a single ROI"""
    if len(neighbors_df) == 0:
        return None
    
    # Create node features
    node_features = []
    for marker in MARKERS:
        if marker in neighbors_df.columns:
            node_features.append(neighbors_df[marker].values)
    
    # Stack features
    x = torch.tensor(np.column_stack(node_features), dtype=torch.float32)
    
    # Create edge index and weights
    edge_index = []
    edge_weights = []
    
    # Add edges between all nodes
    for i in range(len(neighbors_df)):
        for j in range(i + 1, len(neighbors_df)):
            # Calculate distance weight
            dist = neighbors_df.iloc[i]['distance']
            dist_weight = 1.0 / (1.0 + dist)
            
            # Calculate marker compatibility
            m1 = neighbors_df.iloc[i]['marker']
            m2 = neighbors_df.iloc[j]['marker']
            
            # Add bidirectional edges
            edge_index.append([i, j])
            edge_index.append([j, i])
            edge_weights.append(dist_weight)
            edge_weights.append(dist_weight)
    
    if not edge_index:
        return None
    
    edge_index = torch.tensor(edge_index, dtype=torch.long).T
    edge_attr = torch.tensor(edge_weights, dtype=torch.float32).unsqueeze(1)
    
    # Create graph data
    graph = Data(
        x=x,  # Node features
        edge_index=edge_index,  # Edge connections
        edge_attr=edge_attr,  # Edge weights
        pos=torch.tensor(neighbors_df[['x', 'y']].values, dtype=torch.float32),  # Node positions
        marker=neighbors_df['marker'].values  # Node markers
    )
    
    return graph

def calculate_occlusion_distance(x1, y1, z1, x2, y2, z2):
    """Calculate occlusion distance between two 3D points"""
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)

def calculate_intra_edge_weight(node1, node2, max_distance=40):
    """Calculate edge weight for intra-marker connections based on distance and intensity"""
    # Calculate 3D distance
    dist_3d = calculate_occlusion_distance(
        node1['x'], node1['y'], node1['z'],
        node2['x'], node2['y'], node2['z']
    )
    
    # Get intensities for the same marker
    marker = node1['marker']
    intensity1 = node1[marker] if marker in node1 else 0.0
    intensity2 = node2[marker] if marker in node2 else 0.0
    
    # New formula: (intensity1 + intensity2) / sqrt(1 + distance)
    edge_weight = (intensity1 + intensity2) / np.sqrt(1.0 + dist_3d)
    
    return edge_weight, dist_3d

def has_meaningful_interaction(marker1, marker2):
    """Check if two markers have a meaningful biological interaction"""
    # Check if the pair exists in meaningful pairs (in any order)
    return (marker1, marker2) in MEANINGFUL_PAIRS or (marker2, marker1) in MEANINGFUL_PAIRS

def calculate_inter_edge_weight(node1, node2, marker1, marker2):
    """Calculate edge weight for inter-marker connections based on intensity interaction"""
    # Only create edges for meaningful interactions
    if not has_meaningful_interaction(marker1, marker2):
        return 0.0
    
    # Check if this is a same-cell interaction (distance = 0)
    if hasattr(node1, 'distance_3d') and node1['distance_3d'] == 0.0:
        # Same-cell interaction - use higher weight
        intensity1 = node1[marker1] if marker1 in node1 else 0.0
        intensity2 = node1[marker2] if marker2 in node1 else 0.0
        # Higher weight for same-cell interactions
        edge_weight = (intensity1 + intensity2) * 2.0  # Boost same-cell interactions
        return edge_weight
    
    # Calculate 3D distance for different cells
    dist_3d = calculate_occlusion_distance(
        node1['x'], node1['y'], node1['z'],
        node2['x'], node2['y'], node2['z']
    )
    
    # Get intensities for each marker
    intensity1 = node1[marker1] if marker1 in node1 else 0.0
    intensity2 = node2[marker2] if marker2 in node2 else 0.0
    
    # New formula: (intensity1 + intensity2) / sqrt(1 + distance)
    edge_weight = (intensity1 + intensity2) / np.sqrt(1.0 + dist_3d)
    
    return edge_weight

def find_intra_neighbors(roi_row, all_cells_df, marker, max_distance=40):
    """Find intra-marker neighbors within 3D distance"""
    x, y, z = roi_row['x'], roi_row['y'], roi_row['z']
    
    # Find cells of the same marker within maximum distance
    marker_cells = all_cells_df[all_cells_df['marker'] == marker].copy()
    
    # Calculate 3D distances
    marker_cells['distance_3d'] = calculate_occlusion_distance(
        x, y, z,
        marker_cells['x'], marker_cells['y'], marker_cells['z']
    )
    
    # Filter by distance
    neighbors = marker_cells[marker_cells['distance_3d'] <= max_distance].copy()
    
    # Add intensity information
    neighbors['intensity'] = neighbors[marker]
    
    return neighbors

def find_inter_neighbors(roi_row, all_cells_df, marker, fixed_distance=5):
    """Find inter-marker neighbors with meaningful interactions:
    1. First look for meaningful interactions within the same cell (same row)
    2. Then expand radius to find other cells with meaningful interactions"""
    x, y, z = roi_row['x'], roi_row['y'], roi_row['z']
    
    # Step 1: Check for meaningful interactions within the same cell (same row)
    same_cell_interactions = []
    for marker1, marker2 in MEANINGFUL_PAIRS:
        if marker1 == marker:
            # Check if this cell has meaningful expression of marker2
            if marker2 in roi_row and roi_row[marker2] > 0.3:  # Threshold for meaningful expression
                same_cell_interactions.append({
                    'cell_id': roi_row['cell_id'],
                    'marker': marker2,
                    'x': roi_row['x'],
                    'y': roi_row['y'],
                    'z': roi_row['z'],
                    'distance_3d': 0.0,  # Same cell
                    'intensity': roi_row[marker2],
                    'interaction_type': 'same_cell'
                })
                logger.info(f"Found same-cell interaction: {marker} with {marker2} (intensity: {roi_row[marker2]:.3f})")
        elif marker2 == marker:
            # Check if this cell has meaningful expression of marker1
            if marker1 in roi_row and roi_row[marker1] > 0.3:  # Threshold for meaningful expression
                same_cell_interactions.append({
                    'cell_id': roi_row['cell_id'],
                    'marker': marker1,
                    'x': roi_row['x'],
                    'y': roi_row['y'],
                    'z': roi_row['z'],
                    'distance_3d': 0.0,  # Same cell
                    'intensity': roi_row[marker1],
                    'interaction_type': 'same_cell'
                })
                logger.info(f"Found same-cell interaction: {marker} with {marker1} (intensity: {roi_row[marker1]:.3f})")
    
    # Step 2: Find cells of other markers that have meaningful interactions at distance
    meaningful_markers = []
    for marker1, marker2 in MEANINGFUL_PAIRS:
        if marker1 == marker:
            meaningful_markers.append(marker2)
        elif marker2 == marker:
            meaningful_markers.append(marker1)
    
    meaningful_markers = list(set(meaningful_markers))  # Remove duplicates
    
    if not meaningful_markers:
        logger.warning(f"No meaningful interactions found for {marker}")
        # Return same-cell interactions if any
        if same_cell_interactions:
            return pd.DataFrame(same_cell_interactions)
        return pd.DataFrame()  # No meaningful interactions
    
    logger.info(f"Found meaningful markers for {marker}: {meaningful_markers}")
    
    # Find cells of meaningful markers at fixed distance
    other_marker_cells = all_cells_df[all_cells_df['marker'].isin(meaningful_markers)].copy()
    
    # Calculate 3D distances
    other_marker_cells['distance_3d'] = calculate_occlusion_distance(
        x, y, z,
        other_marker_cells['x'], other_marker_cells['y'], other_marker_cells['z']
    )
    
    # Filter by fixed distance
    neighbors = other_marker_cells[other_marker_cells['distance_3d'] <= fixed_distance].copy()
    
    # Add intensity information for each marker
    for marker_name in MARKERS:
        if marker_name in neighbors.columns:
            neighbors[f'{marker_name}_intensity'] = neighbors[marker_name]
    
    # Add interaction type
    neighbors['interaction_type'] = 'nearby_cell'
    
    # Combine same-cell and nearby-cell interactions
    if same_cell_interactions:
        same_cell_df = pd.DataFrame(same_cell_interactions)
        # Add missing columns to match neighbors structure
        for marker_name in MARKERS:
            if marker_name in same_cell_df.columns:
                same_cell_df[f'{marker_name}_intensity'] = same_cell_df[marker_name]
            else:
                same_cell_df[f'{marker_name}_intensity'] = 0.0
        
        # Combine the dataframes
        all_interactions = pd.concat([same_cell_df, neighbors], ignore_index=True)
        logger.info(f"Found {len(same_cell_interactions)} same-cell + {len(neighbors)} nearby-cell interactions for {marker}")
    else:
        all_interactions = neighbors
        logger.info(f"Found {len(neighbors)} nearby-cell interactions for {marker}")
    
    return all_interactions

def get_edge_name(marker1, marker2, edge_type):
    """Get edge name based on markers and edge type"""
    if edge_type == 'intra':
        return f"{marker1}_intra_{marker1}"
    elif edge_type == 'inter':
        # Get interaction name from MEANINGFUL_PAIRS
        for interaction_name, markers in INTERACTIONS.items():
            if marker1 in markers and marker2 in markers:
                return f"{interaction_name}_{marker1}_{marker2}"
        
        # If not found in INTERACTIONS, create a generic name
        return f"inter_{marker1}_{marker2}"
    else:
        return f"unknown_{marker1}_{marker2}"

def build_marker_graph(all_cells_df, marker):
    """Build one main graph for a specific marker with meaningful interactions only"""
    logger.info(f"Building main graph for {marker} with meaningful interactions...")
    
    # Filter cells for this marker
    marker_cells = all_cells_df[all_cells_df['marker'] == marker].copy()
    logger.info(f"Found {len(marker_cells)} cells for {marker}")
    
    if len(marker_cells) == 0:
        logger.warning(f"No cells found for {marker}")
        return None
    
    # Special handling for CD11c (nodes only, no edges)
    if marker == "CD11c":
        logger.info(f"Processing {marker} as nodes only (no edges)")
        return build_cd11c_nodes_only(all_cells_df, marker_cells)
    
    # Create main graph structure
    all_node_features = []
    all_edge_indices = []
    all_edge_weights = []
    all_edge_types = []  # 'intra' or 'inter'
    all_edge_names = []  # Edge names based on interactions
    all_node_ids = []  # Store actual node IDs
    all_edge_info = []  # Store detailed edge information
    subgraph_info = []
    
    node_offset = 0  # To keep track of global node indices
    
    # Process each ROI for this marker
    for idx, roi_row in marker_cells.iterrows():
        logger.info(f"Processing ROI {idx} for {marker}")
        
        # Find intra-marker neighbors (same marker, 25-40 distance)
        intra_neighbors = find_intra_neighbors(roi_row, all_cells_df, marker, max_distance=40)
        logger.info(f"Found {len(intra_neighbors)} intra-neighbors")
        
        # Find inter-marker neighbors (other markers, fixed 5 distance)
        inter_neighbors = find_inter_neighbors(roi_row, all_cells_df, marker, fixed_distance=5)
        logger.info(f"Found {len(inter_neighbors)} inter-neighbors")
        
        # Combine all neighbors for this subgraph
        # Ensure both DataFrames have the same structure before concatenating
        if len(intra_neighbors) > 0 and len(inter_neighbors) > 0:
            # Add missing columns to intra_neighbors to match inter_neighbors
            for col in inter_neighbors.columns:
                if col not in intra_neighbors.columns:
                    if col.endswith('_intensity'):
                        # For intensity columns, use the base marker value
                        base_marker = col.replace('_intensity', '')
                        if base_marker in intra_neighbors.columns:
                            intra_neighbors[col] = intra_neighbors[base_marker]
                        else:
                            intra_neighbors[col] = 0.0
                    elif col == 'interaction_type':
                        intra_neighbors[col] = 'intra'
                    else:
                        intra_neighbors[col] = None
            
            # Add missing columns to inter_neighbors to match intra_neighbors
            for col in intra_neighbors.columns:
                if col not in inter_neighbors.columns:
                    if col == 'intensity':
                        # Use the marker-specific intensity
                        inter_neighbors[col] = inter_neighbors[inter_neighbors['marker']].values
                    else:
                        inter_neighbors[col] = None
            
            # Now concatenate with consistent columns
            all_neighbors = pd.concat([intra_neighbors, inter_neighbors], ignore_index=True)
        elif len(intra_neighbors) > 0:
            all_neighbors = intra_neighbors
        elif len(inter_neighbors) > 0:
            all_neighbors = inter_neighbors
        else:
            all_neighbors = pd.DataFrame()
        
        if len(all_neighbors) == 0:
            continue
        
        # Create node features for this subgraph
        subgraph_features = []
        for marker_name in MARKERS:
            if marker_name in all_neighbors.columns:
                # Fill NaN values with 0
                values = all_neighbors[marker_name].fillna(0.0).values
                subgraph_features.append(values)
            else:
                logger.warning(f"Missing {marker_name} in neighbors for {marker}")
                # Add zeros for missing marker
                subgraph_features.append(np.zeros(len(all_neighbors)))
        
        # Stack features
        subgraph_x = np.column_stack(subgraph_features)
        all_node_features.append(subgraph_x)
        
        # Store node IDs for this subgraph
        subgraph_node_ids = []
        for i, neighbor in all_neighbors.iterrows():
            node_info = {
                'global_id': node_offset + i,
                'cell_id': neighbor['cell_id'],
                'marker': neighbor['marker'],
                'x': neighbor['x'],
                'y': neighbor['y'],
                'z': neighbor['z'],
                'intensities': {marker_name: neighbor[marker_name] for marker_name in MARKERS if marker_name in neighbor}
            }
            subgraph_node_ids.append(node_info)
        all_node_ids.extend(subgraph_node_ids)
        
        # Create edges for this subgraph
        subgraph_edges = []
        subgraph_weights = []
        subgraph_edge_types = []
        subgraph_edge_names = []
        subgraph_edge_details = []
        
        # 1. Intra-marker edges (between same marker cells) - only if more than 1 node
        if len(intra_neighbors) > 1:
            for i in range(len(intra_neighbors)):
                for j in range(i + 1, len(intra_neighbors)):
                    node1 = intra_neighbors.iloc[i]
                    node2 = intra_neighbors.iloc[j]
                    
                    # Calculate intra edge weight
                    edge_weight, dist_3d = calculate_intra_edge_weight(node1, node2, max_distance=40)
                    
                    # Add edge if weight > 0 (meaningful connection)
                    if edge_weight > 0:
                        # Add bidirectional edges with global indices
                        global_i = node_offset + i
                        global_j = node_offset + j
                        
                        subgraph_edges.append([global_i, global_j])
                        subgraph_edges.append([global_j, global_i])
                        subgraph_weights.append(edge_weight)
                        subgraph_weights.append(edge_weight)
                        subgraph_edge_types.append('intra')
                        subgraph_edge_types.append('intra')
                        
                        # Get edge name
                        edge_name = get_edge_name(node1['marker'], node2['marker'], 'intra')
                        subgraph_edge_names.append(edge_name)
                        subgraph_edge_names.append(edge_name)
                        
                        # Store detailed edge information
                        edge_info = {
                            'edge_id': len(all_edge_info),
                            'edge_name': edge_name,
                            'source_global_id': global_i,
                            'target_global_id': global_j,
                            'source_cell_id': node1['cell_id'],
                            'target_cell_id': node2['cell_id'],
                            'source_marker': node1['marker'],
                            'target_marker': node2['marker'],
                            'edge_type': 'intra',
                            'weight': edge_weight,
                            'distance_3d': dist_3d,
                            'source_intensity': node1[node1['marker']],
                            'target_intensity': node2[node2['marker']]
                        }
                        subgraph_edge_details.append(edge_info)
                        
                        # Add reverse edge info
                        edge_info_reverse = edge_info.copy()
                        edge_info_reverse['edge_id'] = len(all_edge_info) + 1
                        edge_info_reverse['source_global_id'] = global_j
                        edge_info_reverse['target_global_id'] = global_i
                        edge_info_reverse['source_cell_id'] = node2['cell_id']
                        edge_info_reverse['target_cell_id'] = node1['cell_id']
                        edge_info_reverse['source_marker'] = node2['marker']
                        edge_info_reverse['target_marker'] = node1['marker']
                        edge_info_reverse['source_intensity'] = node2[node2['marker']]
                        edge_info_reverse['target_intensity'] = node1[node1['marker']]
                        subgraph_edge_details.append(edge_info_reverse)
        
        # 2. Inter-marker edges (between different marker cells) - only meaningful interactions
        for i in range(len(intra_neighbors)):
            for j in range(len(inter_neighbors)):
                node1 = intra_neighbors.iloc[i]
                node2 = inter_neighbors.iloc[j]
                marker1 = node1['marker']
                marker2 = node2['marker']
                
                # Only create edge if there's a meaningful interaction
                if has_meaningful_interaction(marker1, marker2):
                    # Calculate inter edge weight
                    edge_weight = calculate_inter_edge_weight(node1, node2, marker1, marker2)
                    
                    # Add edge if weight > 0 (meaningful connection)
                    if edge_weight > 0:
                        # Add bidirectional edges with global indices
                        global_i = node_offset + i
                        global_j = node_offset + len(intra_neighbors) + j
                        
                        subgraph_edges.append([global_i, global_j])
                        subgraph_edges.append([global_j, global_i])
                        subgraph_weights.append(edge_weight)
                        subgraph_weights.append(edge_weight)
                        subgraph_edge_types.append('inter')
                        subgraph_edge_types.append('inter')
                        
                        # Get edge name
                        edge_name = get_edge_name(marker1, marker2, 'inter')
                        subgraph_edge_names.append(edge_name)
                        subgraph_edge_names.append(edge_name)
                        
                        # Determine interaction type and distance
                        interaction_type = get_interaction_type(marker1, marker2)
                        if hasattr(node2, 'interaction_type') and node2['interaction_type'] == 'same_cell':
                            distance_3d = 0.0  # Same cell interaction
                        else:
                            distance_3d = calculate_occlusion_distance(
                                node1['x'], node1['y'], node1['z'],
                                node2['x'], node2['y'], node2['z']
                            )
                        
                        # Store detailed edge information
                        edge_info = {
                            'edge_id': len(all_edge_info),
                            'edge_name': edge_name,
                            'source_global_id': global_i,
                            'target_global_id': global_j,
                            'source_cell_id': node1['cell_id'],
                            'target_cell_id': node2['cell_id'],
                            'source_marker': node1['marker'],
                            'target_marker': node2['marker'],
                            'edge_type': 'inter',
                            'weight': edge_weight,
                            'distance_3d': distance_3d,
                            'source_intensity': node1[marker1],
                            'target_intensity': node2[marker2],
                            'interaction_type': interaction_type,
                            'is_same_cell': hasattr(node2, 'interaction_type') and node2['interaction_type'] == 'same_cell'
                        }
                        subgraph_edge_details.append(edge_info)
                        
                        # Add reverse edge info
                        edge_info_reverse = edge_info.copy()
                        edge_info_reverse['edge_id'] = len(all_edge_info) + 1
                        edge_info_reverse['source_global_id'] = global_j
                        edge_info_reverse['target_global_id'] = global_i
                        edge_info_reverse['source_cell_id'] = node2['cell_id']
                        edge_info_reverse['target_cell_id'] = node1['cell_id']
                        edge_info_reverse['source_marker'] = node2['marker']
                        edge_info_reverse['target_marker'] = node1['marker']
                        edge_info_reverse['source_intensity'] = node2[marker2]
                        edge_info_reverse['target_intensity'] = node1[marker1]
                        subgraph_edge_details.append(edge_info_reverse)
        
        all_edge_indices.extend(subgraph_edges)
        all_edge_weights.extend(subgraph_weights)
        all_edge_types.extend(subgraph_edge_types)
        all_edge_names.extend(subgraph_edge_names)
        all_edge_info.extend(subgraph_edge_details)
        
        # Store subgraph information
        subgraph_info.append({
            'roi_id': roi_row['cell_id'],
            'center': [roi_row['x'], roi_row['y'], roi_row['z']],
            'num_intra_nodes': len(intra_neighbors),
            'num_inter_nodes': len(inter_neighbors),
            'num_total_nodes': len(all_neighbors),
            'num_edges': len(subgraph_edges),
            'start_node': node_offset,
            'end_node': node_offset + len(all_neighbors) - 1,
            'node_ids': [node['cell_id'] for node in subgraph_node_ids],
            'edge_details': subgraph_edge_details
        })
        
        node_offset += len(all_neighbors)
    
    if len(all_node_features) == 0:
        logger.warning(f"No valid subgraphs found for {marker}")
        return None
    
    # Combine all features and edges
    x = torch.tensor(np.vstack(all_node_features), dtype=torch.float32)
    
    if len(all_edge_indices) > 0:
        edge_index = torch.tensor(all_edge_indices, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(all_edge_weights, dtype=torch.float32).unsqueeze(1)
    else:
        # Create empty edge tensors if no edges
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 1), dtype=torch.float32)
    
    # Create main graph
    main_graph = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr
    )
    
    # Add metadata with complete information
    main_graph.metadata = {
        'marker': marker,
        'num_nodes': len(x),
        'num_edges': len(all_edge_weights),
        'num_subgraphs': len(subgraph_info),
        'edge_types': all_edge_types,
        'edge_names': all_edge_names,  # Edge names based on interactions
        'nodes': all_node_ids,  # Complete node information
        'edges': all_edge_info,  # Complete edge information
        'subgraphs': subgraph_info
    }
    
    # Count edge types and names
    intra_edges = sum(1 for edge_type in all_edge_types if edge_type == 'intra')
    inter_edges = sum(1 for edge_type in all_edge_types if edge_type == 'inter')
    
    # Count unique edge names
    unique_edge_names = list(set(all_edge_names))
    edge_name_counts = {}
    for name in unique_edge_names:
        edge_name_counts[name] = all_edge_names.count(name)
    
    logger.info(f"Created main graph for {marker}:")
    logger.info(f"  - Total nodes: {len(x)}")
    logger.info(f"  - Total edges: {len(all_edge_weights)}")
    logger.info(f"  - Intra edges: {intra_edges}")
    logger.info(f"  - Inter edges: {inter_edges}")
    logger.info(f"  - Number of subgraphs: {len(subgraph_info)}")
    logger.info(f"  - Edge names: {edge_name_counts}")
    
    return main_graph

def build_cd11c_nodes_only(all_cells_df, cd11c_cells):
    """Build graph for CD11c with nodes only (no edges)"""
    logger.info("Building CD11c graph with nodes only")
    
    # Create node features for all CD11c cells
    node_features = []
    for marker_name in MARKERS:
        if marker_name in cd11c_cells.columns:
            node_features.append(cd11c_cells[marker_name].values)
    
    # Stack features
    x = torch.tensor(np.column_stack(node_features), dtype=torch.float32)
    
    # Create empty edge tensors (no edges for CD11c)
    edge_index = torch.empty((2, 0), dtype=torch.long)
    edge_attr = torch.empty((0, 1), dtype=torch.float32)
    
    # Store complete node information
    all_node_ids = []
    for i, cell in cd11c_cells.iterrows():
        node_info = {
            'global_id': i,
            'cell_id': cell['cell_id'],
            'marker': cell['marker'],
            'x': cell['x'],
            'y': cell['y'],
            'z': cell['z'],
            'intensities': {marker: cell[marker] for marker in MARKERS if marker in cell}
        }
        all_node_ids.append(node_info)
    
    # Create graph
    main_graph = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr
    )
    
    # Add metadata with complete information
    main_graph.metadata = {
        'marker': 'CD11c',
        'num_nodes': len(x),
        'num_edges': 0,
        'num_subgraphs': 1,
        'edge_types': [],
        'edge_names': [],  # No edge names for CD11c
        'nodes': all_node_ids,  # Complete node information
        'edges': [],  # No edges for CD11c
        'subgraphs': [{
            'roi_id': 'CD11c_all',
            'center': [0, 0, 0],  # Not applicable for CD11c
            'num_intra_nodes': len(cd11c_cells),
            'num_inter_nodes': 0,
            'num_total_nodes': len(cd11c_cells),
            'num_edges': 0,
            'start_node': 0,
            'end_node': len(cd11c_cells) - 1,
            'node_ids': [node['cell_id'] for node in all_node_ids],
            'edge_details': []
        }]
    }
    
    logger.info(f"Created CD11c graph: {len(x)} nodes, 0 edges")
    
    return main_graph

def get_interaction_type(marker1, marker2):
    """Get the type of biological interaction between two markers"""
    # First check if it's a known interaction from INTERACTIONS
    for interaction_name, markers in INTERACTIONS.items():
        if marker1 in markers and marker2 in markers:
            return interaction_name
    
    # If not found in INTERACTIONS, check if it's a meaningful pair
    if has_meaningful_interaction(marker1, marker2):
        return f"inter_{marker1}_{marker2}"
    
    return "unknown"

def process_rois():
    """Process all ROIs and build main graphs for each marker"""
    logger.info("Starting ROI processing for main marker graphs...")
    
    # Load the normalized cell features
    input_file = os.path.join(OUTPUT_DIR, 'Norm_all_cell_features.csv')
    if not os.path.exists(input_file):
        logger.error(f"Input file not found: {input_file}")
        return
    
    logger.info(f"Loading cell features from {input_file}")
    all_cells_df = pd.read_csv(input_file)
    logger.info(f"Loaded {len(all_cells_df)} cells")
    
    # Create output directory for main graphs
    main_graphs_dir = os.path.join(OUTPUT_DIR, 'main_marker_graphs')
    os.makedirs(main_graphs_dir, exist_ok=True)
    
    # Build one main graph for each marker
    for marker in MARKERS:
        logger.info(f"\n{'='*50}")
        logger.info(f"Processing marker: {marker}")
        logger.info(f"{'='*50}")
        
        # Build main graph for this marker
        main_graph = build_marker_graph(all_cells_df, marker)
        
        if main_graph is not None:
            # Save the main graph
            output_file = os.path.join(main_graphs_dir, f'{marker}_main_graph.pt')
            torch.save(main_graph, output_file)
            logger.info(f"Saved main graph for {marker} to {output_file}")
            
            # Print summary statistics
            metadata = main_graph.metadata
            logger.info(f"Summary for {marker}:")
            logger.info(f"  - Total nodes: {metadata['num_nodes']}")
            logger.info(f"  - Total edges: {metadata['num_edges']}")
            logger.info(f"  - Number of subgraphs: {metadata['num_subgraphs']}")
            
            # Print subgraph details
            logger.info(f"  - Subgraphs:")
            for i, subgraph in enumerate(metadata['subgraphs'][:5]):  # Show first 5
                logger.info(f"    {i+1}. ROI {subgraph['roi_id']}: {subgraph['num_intra_nodes']} intra-nodes, {subgraph['num_inter_nodes']} inter-nodes, total {subgraph['num_total_nodes']} nodes")
            if len(metadata['subgraphs']) > 5:
                logger.info(f"    ... and {len(metadata['subgraphs']) - 5} more subgraphs")
        else:
            logger.warning(f"Failed to create main graph for {marker}")
    
    logger.info(f"\n{'='*50}")
    logger.info("Main marker graphs processing completed!")
    logger.info(f"Output directory: {main_graphs_dir}")
    logger.info(f"{'='*50}")

if __name__ == "__main__":
    process_rois() 