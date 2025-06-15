## 1- create_segmentation1.py 



## 2- merge_features2.py
      :Loaded 234 cells from CD31
      :Loaded 866 cells from CD20
      :Loaded 84 cells from CD11b
      :Loaded 29 cells from CD4
      :Loaded 29 cells from CD11c
      :Loaded 227 cells from Catalase
      :Saved 1469 total cells to output/All_cell_features.csv
      :Normalized CD31: max value was 17716.25
      :Normalized CD20: max value was 27683.00
      :Normalized CD11b: max value was 26668.38
      :Normalized CD4: max value was 26340.80
      :Normalized CD11c: max value was 15496.33
      :Normalized Catalase: max value was 27347.50
      :Saved normalized features to output/Norm_all_cell_features.csv
      :
      Cell counts by marker:
      :CD31: 234 cells
      :CD20: 866 cells
      :CD11b: 84 cells
      :CD4: 29 cells
      :CD11c: 29 cells
      :Catalase: 227 cells
      :
      :CD20: 866 cells
      :CD11b: 84 cells
      :CD4: 29 cells
      :CD11c: 29 cells
      :Catalase: 227 cells
      :
      :CD20: 866 cells
      :CD11b: 84 cells
      :CD4: 29 cells
      :CD11c: 29 cells
      :Catalase: 227 cells
      :
      :CD20: 866 cells
      :CD11b: 84 cells
      :CD4: 29 cells
      :CD11c: 29 cells
      :Catalase: 227 cells
      :
      :CD20: 866 cells
      :CD11b: 84 cells
      :CD4: 29 cells
      :CD11c: 29 cells
      :Catalase: 227 cells
      :
      :CD20: 866 cells
      :CD11b: 84 cells
      :CD4: 29 cells
      :CD11c: 29 cells
      :Catalase: 227 cells
      :
      :CD20: 866 cells
      :CD11b: 84 cells
      :CD4: 29 cells
      :CD11c: 29 cells
      :Catalase: 227 cells
      :
      :CD20: 866 cells
      :CD11b: 84 cells
      :CD4: 29 cells
      :CD11c: 29 cells


## 3  build_graph_gat.py
           #//  Location: output/roi_graphs_{marker}.pt
      1. Graph Structure
      Each graph represents a Region of Interest (ROI)
      Specific to one marker (CD31, CD20, etc.)
      Built using PyTorch Geometric's Data class
      2. Nodes (Cells)
      Node Features (x)
      Normalized marker values
      Shape: [num_nodes, num_markers]
      Node Positions (pos)
      x, y coordinates
      Shape: [num_nodes, 2]
      Node Markers (marker)
      Type of marker for each cell
      Shape: [num_nodes]
      3. Edges (Connections)
      Edge Index (edge_index)
      Shows node connections
      Shape: [2, num_edges]
      Example: [[0,1], [1,0]]
      Edge Weights (edge_attr)
      Based on distance: 1.0 / (1.0 + distance)
      Shape: [num_edges, 1]
      4. Graph Building Steps
      Create node features from marker values
      Calculate distances between cells
      Create bidirectional edges
      Assign edge weights based on distances
      Combine into PyTorch Geometric Data object
      5. Neighbor Finding
      Find cells within radius (30 units)
      Calculate actual distances
      Filter by distance threshold
      Create neighborhood graph
      6. GAT Model Structure
      First Layer
      Input: Node features + Edge attributes
      Output: Hidden representations
      Second Layer
      Input: Hidden representations
      Output: Final node embeddings
      7. Output Files
      Per Marker
      File: roi_graphs_{marker}.pt
      Contains:
      List of graphs
      GAT model state
      8. Graph Statistics
      Number of nodes
      Number of edges
      Feature dimensions
      Edge attribute dimensions
      9. Learning Capabilities
      Cell-to-cell interactions
      Marker-specific patterns
      Spatial relationships
      Marker importance in ROIs
  //#  1. Generated Files
                  CD31: roi_graphs_CD31.pt (234 graphs)
                  CD20: roi_graphs_CD20.pt (861 graphs)
                  CD11b: roi_graphs_CD11b.pt (83 graphs)
                  CD4: roi_graphs_CD4.pt (27 graphs)
                  CD11c: roi_graphs_CD11c.pt (24 graphs)
                  Catalase: roi_graphs_Catalase.pt (214 graphs)

  # 4. train_gat4
   # Graph Attention Network (GAT) Architecture for Spatial Feature Learning
# Enhanced Graph Attention Network for Spatial Biology

            ## Model Architecture
            Our model combines graph attention mechanisms with biological interaction knowledge:

            1. **Base Architecture**:
            - Input: 6 marker features (CD31, CD20, CD11b, CD4, CD11c, Catalase)
            - Two GAT layers: 64→32 dimensions with 4 attention heads
            - Output: 6-dimensional reconstruction

            2. **Biological Interaction Module**:
            - Incorporates known marker interactions:
            * T-cell entry site (CD31-CD4)
            * Inflammatory zone (CD11b-CD20)
            * Oxidative stress niche (CD11b-Catalase)
            * B-cell infiltration (CD20-CD31)
            * Dendritic signal (CD11c)
            - Multi-head attention layers for each interaction
            - Interaction-aware feature enhancement

                  ## Key Innovations
                  1. **Dual Attention Mechanism**:
                  - Graph attention for spatial relationships
                  - Interaction attention for biological patterns

                  2. **Self-supervised Learning**:
                  - Feature reconstruction objective
                  - Preserves biological interaction patterns

            ## Training Process
            - Objective: Minimize reconstruction error
            - Optimizer: Adam (lr=0.001)
            - Batch size: 32
            - Epochs: 100 per marker

            ## Evaluation Metrics
            - RMSE, MAE, R² for reconstruction quality
            - Interaction pattern preservation

            Project Overview:
            We're working on a spatial transcriptomics analysis project using a GAT (Graph Attention Network) model
            The project involves processing biological data with multiple markers (CD3, CD8, CD20, etc.)
            We're using a combination of feature extraction and graph-based learning approaches
            Recent Progress:
            Successfully created segmentation files for various markers
            Implemented a feature extraction pipeline
            Developed a GAT model training script
            Fixed several issues including:
            Tensor size mismatches
            Missing file dependencies
            Tkinter-related errors
            Current State:
            The training script has been modified to:
            Add a final layer to match output dimensions
            Improve loss calculation explanations
            Handle data processing more efficiently
            The script has been executed successfully
            Next Steps:
            We need to implement proper logging of training and test accuracy
            The results should be printed to the console for monitoring
            We may need to add visualization capabilities for the results



  # 5.evaluate_gat5

      Marker      Train    Test    TrLoss    TeLoss    TrRMSE    TeRMSE    TrMAE    TeMAE    TrR2    TeR2    TrAcc    TeAcc    TrRec    TeRec    TrF1    TeF1
      --------  -------  ------  --------  --------  --------  --------  -------  -------  ------  ------  -------  -------  -------  -------  ------  ------
      CD31          198      36     0.013     0.014     0.546     0.575    0.299     0.33  -0.778  -1.072    0.701     0.67    0.701     0.67   0.724   0.698
      CD20          731     130      0.01      0.01     0.558     0.579    0.311    0.336  -0.901  -1.018    0.689    0.664    0.689    0.664   0.714   0.687
      CD11b          70      13      0.01     0.013     0.611     0.643    0.374    0.413  -1.083  -1.296    0.626    0.587    0.626    0.587   0.655   0.616
      CD4            22       5     0.015     0.013     0.609     0.717    0.371    0.514   -1.35   -2.14    0.629    0.485    0.629    0.485   0.664   0.534
      CD11c          20       4     0.015     0.013     0.636     0.615    0.405    0.378  -1.659  -1.407    0.595    0.622    0.595    0.622   0.639   0.661
      Catalase      181      33     0.012     0.012      0.55     0.569    0.303    0.324  -0.857  -0.973    0.697    0.676    0.697    0.676   0.723   0.706

# Extract_ROI.py
      Inputs:
      ROI graph files for each marker in the output folder named roi_graphs_{marker}.pt
      Trained GAT model for each marker in the output folder named best_model_{marker}.pt
      Method:
      For each interaction in INTERACTIONS, loads graphs for the required markers
      Uses the GAT model to extract latent features (embeddings)
      Computes interaction scores for each ROI
      Outputs:
      JSON files for each interaction in the output folder named extraction_roi_{interaction_name}.json
      Each file contains position (x, y, z) and score of each ROI sorted by score
      Purpose:
      Identifies important interaction regions between markers in spatial data
      Ranks ROIs based on their interaction significance


 # 7-python generate_vitnesse_config_sdk.py
      
      ==================================================
      VITNESSE CONFIGURATION SUMMARY
      ==================================================

      B-cell infiltration:
      ------------------------------
      Number of ROIs: 10
      Score range: 0.2654 - 0.1316

      Dendritic signal:
      ------------------------------
      Number of ROIs: 10
      Score range: 0.2557 - 0.0401

      Inflammatory zone:
      ------------------------------
      Number of ROIs: 10
      Score range: 0.2653 - 0.0840

      Oxidative stress niche:
      ------------------------------
      Number of ROIs: 10
      Score range: 0.3168 - 0.1462

      T-cell entry site:
      ------------------------------
      Number of ROIs: 10
      Score range: 0.2117 - 0.1597

      ==================================================
      Generated 50 ROIs in GeoJSON format