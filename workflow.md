# SpaFGAN Workflow Documentation

## Overview
This document outlines the workflow for processing and analyzing 3D multi-channel microscopy data, focusing on cell segmentation and feature extraction.

## Project Structure
```
SpaFGAN/
├── backend/
│   ├── input/          # Input data directory
│   ├── output/         # Output results directory
│   ├── download.py     # Data download script
│   ├── segment.py      # Cell segmentation script
│   ├── extract_features.py  # Feature extraction script
│   ├── build_graph.py  # Spatial graph construction script
│   ├── install_dependencies.py  # Dependency installation script
│   └── train_spafgan.py  # SpaFGAN training script (coming soon)
└── frontend/          # Frontend visualization (not covered in this documentation)
```

## Step-by-Step Process

### 1. Data Download (`download.py`)
- **Purpose**: Download specific channels from the Zarr dataset
- **Channels Selected**:
  - CD31 (index 19)
  - CD20 (index 27)
  - CD11b (index 37)
  - CD4 (index 25)
  - Catalase (index 59)
  - CD11c (index 41)
- **Process**:
  - Connects to S3 storage
  - Downloads selected channels
  - Saves data in Zarr format with compression
  - Stores metadata about the channels
- **Output**: `backend/input/selected_channels.zarr`

### 2. Cell Segmentation (`segment.py`)
- **Purpose**: Segment cells using CD31 channel
- **Process**:
  - Loads 6-channel Zarr data
  - Applies Otsu thresholding
  - Performs 3D connected components analysis
  - Removes small objects
  - Labels individual cells
- **Parameters**:
  - Minimum cell size: 50 voxels
  - Compression: zstd (level 5)
- **Output**: `backend/input/segmentation.zarr`

### 3. Feature Extraction (`extract_features.py`)
- **Purpose**: Extract quantitative features from segmented cells
- **Features Extracted**:
  - 3D centroid position (z, y, x)
  - Mean intensity for each channel:
    - CD31
    - CD20
    - CD11b
    - CD4
    - CD11c
    - Catalase
- **Process**:
  - Loads segmented data
  - Calculates centroids
  - Computes mean intensities
  - Generates summary statistics
- **Output**: `backend/output/cell_features.csv`

### 4. Spatial Graph Construction (`build_graph.py`)
- **Purpose**: Build a spatial graph representation of cells
- **Process**:
  - Loads cell features from CSV
  - Creates nodes for each cell with:
    - Features: 6-channel biomarker intensities
    - Attributes: 3D position and cell ID
  - Connects nearby cells using radius-based neighborhood
  - Computes graph statistics
- **Parameters**:
  - Neighborhood radius: 30.0 units
- **Output**: `backend/output/spatial_graph.pkl`
- **Graph Structure**:
  - Nodes: Cells with biomarker features
  - Edges: Spatial connections between nearby cells
  - Node attributes: Position (z,y,x) and cell ID
  - Edge attributes: None (binary connections)

### 5. SpaFGAN Training (Coming Soon)
- **Purpose**: Train a Graph Attention Network for immune ROI prediction
- **Components**:
  - PyTorch-based Graph Attention Network
  - Node scoring head for immune ROI prediction
  - Training loop (supervised or contrastive)
- **Dependencies**:
  - PyTorch
  - PyTorch Geometric
  - Additional ML libraries
- **Process**:
  1. Install dependencies:
     ```bash
     python backend/install_dependencies.py
     ```
  2. Train model (coming soon):
     ```bash
     python backend/train_spafgan.py
     ```

### 6. Extract ROIs (Immune ROIs)
- **Purpose**: Extract immune ROIs from the trained model
- **Process**:
  - Load the trained model and spatial graph
  - Compute ROI scores for each cell
  - Threshold the scores (e.g., 0.7)
  - Extract connected components from cells with high scores
  - Save cell information for each ROI in `backend/output/roi_cells.csv`
  - Each row contains: ROI ID, cell ID, coordinates (x, y, z), and model score.

### 7.1 Convert Cells + ROIs to AnnData (Zarr)
- **Purpose**: Convert cell features and ROI information to AnnData format for visualization with Vitessce
- **Process**:
  - Load cell features from `backend/output/cell_features.csv`
  - Load ROI information from `backend/output/roi_cells.csv`
  - Merge ROI scores and IDs into cell data
  - Create AnnData object with features, metadata, and spatial coordinates
  - Save AnnData as Zarr in `backend/output/cells.zarr`
- **Dependencies**:
  - anndata
  - zarr
- **Installation**:
  ```bash
  pip install anndata zarr
  ```
- **Output**: `backend/output/cells.zarr`

## Data Formats

### Input Data
- Original data: Multi-channel 3D microscopy data
- Format: Zarr array
- Shape: (1, 70, 194, 688, 1363)
- Chunk size: (1, 1, 1, 688, 1024)

- **Processing Steps**:
  1. Load Zarr data
  2. Apply coordinate transformations:
     ```
     X_scaled = X * (10908/1363)
     Y_scaled = Y * (5508/688)
     Y_flipped = 5508 - Y_scaled
     ```
  3. Extract ROI information:
     * Parse GeoJSON features
     * Calculate centroids
     * Extract properties

### 2. Backend Operations
- **API Endpoints**:
  * `/api/roi_shapes`: GET
    - Returns: GeoJSON with ROI data
    - Properties: name, score, interactions

- **Data Management**:
  * Coordinate system handling
  * ROI grouping
  * Interaction type filtering

### 3. Frontend Operations
- **Component Lifecycle**:
  1. Mount
  2. Data fetching
  3. State initialization
  4. User interaction handling
  5. View updates

- **State Management**:
  * Selected interaction types
  * Current ROI index
  * Coordinate values
  * View parameters

### 4. User Interaction Patterns
- **Selection Flow**:
  1. View available interactions
  2. Toggle selections
  3. See filtered ROIs
  4. Navigate results

## Dependencies
- zarr
- numpy
- pandas
- scikit-image
- scipy
- s3fs
- networkx
- scikit-learn
- torch
- torch-geometric
- torch-scatter
- torch-sparse
- anndata

## Usage
1. Download data:
   ```bash
   python backend/download.py
   ```

2. Segment cells:
   ```bash
   python backend/segment.py
   ```

3. Extract features:
   ```bash
   python backend/extract_features.py
   ```

4. Build spatial graph:
   ```bash
   python backend/build_graph.py
   ```

5. Install dependencies for SpaFGAN:
   ```bash
   python backend/install_dependencies.py
   ```

6. Train SpaFGAN (coming soon):
   ```bash
   python backend/train_spafgan.py
   ```

7. Extract ROIs:
   ```bash
   python backend/extract_rois.py
   ```

8. Convert cells + ROIs to AnnData (Zarr):
   ```bash
   pip install anndata zarr
   python backend/convert_to_anndata.py
   ```

## Visualization and Analysis
- Visualize ROIs using tools like Vitessce
- Perform biological analysis on extracted ROIs

## Notes
- Automatic ROI extraction after SpaFGAN training
- Visualization using Vitessce:
  - Load `vitessce_config.json` in Vitessce Viewer
  - View spatial data, cell features, and ROI overlays
  - Interactive exploration of immune ROIs 