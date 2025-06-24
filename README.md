# SpaFGAN: Spatial Feature Graph Attention Network for Biological Data Analysis

This project implements a Graph Attention Network (GAT) based approach for analyzing spatial biological data with multiple cellular markers. The system processes OME-Zarr data, performs segmentation, extracts features, builds local ROI graphs, and generates Vitnesse configurations for visualization.

## Project Structure

```
SpaFGAN/
├── backend/                    # Python backend processing
│   ├── data0.py               # Data loading and preprocessing from OME-Zarr
│   ├── download00.py          # Data download utilities
│   ├── create_segmentation1.py # Cell segmentation and mask generation
│   ├── merge_features2.py     # Feature extraction and merging
│   ├── build_graph_gat3.py    # Local ROI graph construction
│   ├── train_gat4.py          # GAT model training for different markers
│   ├── evaluate_gat5.py       # Model evaluation and performance metrics
│   ├── Extract_ROI6.py        # ROI extraction and scoring system
│   ├── generate_vitnesse_config_sdk7.py # Vitnesse visualization config
│   ├── surrogate_labeling.py  # Surrogate labeling for training
│   ├── models/                # Trained model files (*.pt)
│   ├── output/                # Generated outputs and results
│   └── graph_analysis/        # Graph statistics and visualizations
├── frontend/                  # React frontend for visualization
│   ├── src/
│   │   ├── components/        # React components (Mainview, ROISelector, etc.)
│   │   └── App.jsx           # Main application component
│   └── package.json
└── README.md
```

## Setup

1.  **Backend:**
    *   Navigate to `backend/` directory
    *   Create conda environment: `conda create -n SPA python=3.8`
    *   Activate environment: `conda activate SPA`
    *   Install dependencies: `pip install -r requirements.txt`
    *   Ensure OME-Zarr data is available in `selected_channels.zarr`

2.  **Frontend:**
    *   Navigate to `frontend/` directory
    *   Install Node.js dependencies: `npm install`
    *   Ensure all required packages are installed

## Running the Application

1.  **Start Backend:**
    *   Open a terminal.
    *   Navigate to `backend/`.
    *   Activate conda environment: `conda activate SPA`
    *   Run processing pipeline in sequence:
        ```bash
        python data0.py                    # Data loading and preprocessing
        python create_segmentation1.py     # Cell segmentation
        python merge_features2.py          # Feature extraction
        python build_graph_gat3.py         # Graph construction
        python train_gat4.py               # Model training
        python evaluate_gat5.py            # Model evaluation
        python Extract_ROI6.py             # ROI extraction and scoring
        python generate_vitnesse_config_sdk7.py  # Vitnesse config generation
        ```

2.  **Start Frontend:**
    *   Open a *second* terminal.
    *   Navigate to `frontend/`.
    *   Run the development server: `npm run dev`
    *   The frontend will likely run on `http://localhost:5173` (check terminal output).

3.  **View:**
    *   Open your web browser and navigate to the frontend URL (e.g., `http://localhost:5173`). 
    *   The application will display spatial data visualization with ROI overlays
    *   Use the ROI selector to filter and view different cellular interactions

## Evaluation and Results

### Model Performance
- **CD4 (T-cell entry site)**: GAT model trained for T-cell interaction analysis
- **CD11b (Inflammatory zone)**: GAT model for inflammatory response detection
- **Catalase (Oxidative stress niche)**: GAT model for oxidative stress analysis
- **CD20 (B-cell infiltration)**: GAT model for B-cell infiltration patterns
- **CD11c (Dendritic signal)**: MLP model for dendritic cell signaling

### ROI Scoring System
- **Intensity Score**: Based on marker expression levels (nodes × average intensity)
- **Attention Score**: Based on graph attention weights from trained models
- **Combined Score**: Weighted combination of intensity and attention scores
- **Top ROIs**: Top 10 ROIs per interaction type saved with positions and scores

### Output Files
- `vitnesse_config.json`: Vitnesse visualization configuration
- `roi_shapes.geojson`: ROI shapes in GeoJSON format for mapping
- `top_roi_scores_*.json`: Top ROI scores for each cellular interaction
- `gat_evaluation_results.json`: Model evaluation metrics and performance
- `best_model_*.pt`: Trained model files for each marker type

### Visualization Features
- Interactive spatial visualization of ROIs
- Color-coded cellular interactions
- Score-based filtering and highlighting
- Multi-marker overlay capabilities
- Real-time ROI selection and analysis

### 1-surrogate_labeling.py
- Implements surrogate labeling for training data generation
- Creates synthetic labels for unsupervised learning scenarios
- Supports the training pipeline for GAT models
