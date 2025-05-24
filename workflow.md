# Workflow Documentation

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

## Data Formats

### Input Data
- Original data: Multi-channel 3D microscopy data
- Format: Zarr array
- Shape: (1, 70, 194, 688, 1363)
- Chunk size: (1, 1, 1, 688, 1024)

### Output Data
1. **Selected Channels**:
   - Format: Zarr
   - Shape: (1, 6, 194, 688, 1363)
   - Compression: zstd

2. **Segmentation**:
   - Format: Zarr
   - Shape: (194, 688, 1363)
   - Contains labeled cells

3. **Cell Features**:
   - Format: CSV
   - Columns:
     - cell_id
     - z, y, x (centroid coordinates)
     - CD31, CD20, CD11b, CD4, CD11c, Catalase (mean intensities)

4. **Spatial Graph**:
   - Format: Pickle
   - Type: networkx.Graph
   - Contains nodes and edges with spatial relationships

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