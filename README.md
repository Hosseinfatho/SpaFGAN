# Camera Position Visualization Project

This project displays OME-Zarr data using Vitessce, with a Flask backend and a React frontend.

## Project Structure

```
/
├── backend/           # Python Flask backend
│   ├── CameraPosition/  # Python Virtual Environment (created by user)
│   ├── static/        # Static files (if served by Flask - currently unused)
│   ├── templates/     # HTML templates (if served by Flask - currently unused)
│   ├── requirements.txt # Backend Python dependencies
│   └── server.py      # Flask application
├── frontend/          # React frontend
│   ├── node_modules/  # Node.js dependencies (managed by npm)
│   ├── public/        # Static assets for frontend
│   ├── src/           # React source code
│   ├── package.json   # Frontend dependencies and scripts
│   └── ...            # Other frontend config files (vite.config.js, etc.)
└── README.md        # This file
```

## Setup

1.  **Backend:**
    *   Navigate to the `backend` directory: `cd backend`
    *   Create/ensure the Python virtual environment exists (e.g., `python -m venv CameraPosition`)
    *   Activate the virtual environment: `.\CameraPosition\Scripts\activate` (Windows)
    *   Install dependencies: `pip install -r requirements.txt`

2.  **Frontend:**
    *   Navigate to the `frontend` directory: `cd frontend`
    *   Install dependencies: `npm install`

## Running the Application

1.  **Start Backend:**
    *   Open a terminal.
    *   Navigate to `backend/`.
    *   Activate the virtual environment: `.\CameraPosition\Scripts\activate`
    *   Run the server: `python server.py`
    *   The backend will run on `http://127.0.0.1:5000`.

2.  **Start Frontend:**
    *   Open a *second* terminal.
    *   Navigate to `frontend/`.
    *   Run the development server: `npm run dev`
    *   The frontend will likely run on `http://localhost:5173` (check terminal output).

3.  **View:**
    *   Open your web browser and navigate to the frontend URL (e.g., `http://localhost:5173`). 


1.  extract_features.py       → CSV with cell x, y, z and markers
2.  train_spafgan.py          → Trained SpaFGAN model
3.  segment.py                → Labeled cell volume
4.  build_graph.py            → Graph of cells and features
5.  extract_rois.py           → ROIs + interactions
6.  generate_roi_shapes.py    → (optional override for shapes)
7.  convert_to_anndata.py     → (optional for omics/scanpy)
8. generate_vitessce_config_sdk → dynamic Visualization setup







General description of the extract_rois.py file:
This script uses spatial graphs and a trained SpaFGAN model to extract regions of interest (ROI). The main steps are as follows:
Structure and steps:
Define markers and parameters
Markers: CD31, CD11b, CD11c (each with radius and minimum number of nodes)
Number of output ROIs: 20
Function process_marker
Input: marker name, model path, graph path, cell feature path, output path
Load graph and features
Load model and prepare on GPU/CPU
Prepare node features and convert to PyG format
Predict the score of each node with the model
Select nodes with the highest score (top N)
Build subgraph and find connected components
Calculate the score of each component and select the best ones
Save results in CSV file
Function main
Creates the path to the required files for each marker
If all files are available, performs ROI extraction





1-We have: cell_features_{marker}.csv (from extract_features.py)
        Required Order:
        First, we need to train the model to get spafgan_model.pt
        Then use that model in segment.py to create roi_cells_{marker}.csv
        Finally, use both files in build_graph.py to create the spatial graphs

2-What is SpaFGAN model
    SpaFGAN (Spatial Feature Graph Attention Network) is a model designed to:
    Analyze Spatial Relationships: Understand how cells are organized in 3D space
    Detect Important Regions: Identify biologically significant areas in the tissue
    Learn Cell Interactions: Understand how different cell types interact with each other

    How it Helps Us:
    ROI Detection:
    Identifies Regions of Interest (ROIs) in the tissue
    Finds areas where specific markers are highly expressed
    Example: Finding areas with high CD31 expression (vascular regions)
    Cell Interaction Analysis:
    Discovers how different cell types interact
              INTERACTIONS = {
         "T-cell entry site": {"CD31": "high", "CD4": "high"},
         "Inflammatory zone": {"CD11b": "high", "CD20": "high"},
         "Oxidative stress niche": {"CD11b": "high", "Catalase": "high"}
     }
        we have two separate models for CD31 and CD11b. The training results show:
                CD31 Model:
                Final accuracy: 67.33%
                Final loss: 0.6142
                Training cells: 188 cells
                CD11b Model:
                Final accuracy: 98.71%
                Final loss: 0.0817
                Training cells: 6 cells
                Key differences:
                Training Data:
                CD31: Uses its own data
                CD11b: Uses both CD11b and CD31 data for learning

  3- The segment.py script is responsible for segmenting and identifying Regions of Interest (ROIs) using the trained SpaFGAN models. Here's what it does step by step:
    
    *Main Processing Function (process_marker):
        Loads cell features from CSV
        Normalizes features using MinMaxScaler
        Creates spatial graph using k-nearest neighbors
        Loads the trained SpaFGAN model
        Makes predictions for each cell
        Applies thresholds to identify ROIs
        Detects biological interactions
        Saves results to CSV



4-the build_graph.py script in English:
This script constructs a spatial graph of cells and Regions of Interest (ROIs). Here are the main components:
Marker Definitions:
            Primary Markers: CD31 and CD11b (ROI centers)
            Secondary Markers: CD4, CD20, and Catalase (connecting cells)
            Edge Weight Calculation:
            The calculate_edge_weight function computes weights based on:
            Distance between cells (closer = higher weight)
            Cell types (different connections have different weights)
            For example, connection between CD31 and CD11b has weight 1.5
            Spatial Graph Construction:
            The build_spatial_graph function for each primary marker:
            Loads ROI and cell data
            Creates a new graph
            Adds ROI nodes
            Adds cell nodes
            Connects cells within specified radius
            Connects nearby cells to each other
            Key Parameters:
            Connection radius for CD11b: 150.0 units
            Connection radius for CD31: 50.0 units
            Each cell connects to 5 nearest neighbors
            Output:
            The constructed graph is saved in spatial_graph_{marker}.pkl
            Contains information about:
            Spatial positions of cells
            Marker values
            Connection weights



5- Extract_roi
        | Input | Output |
        |-----------------------------------|-----------------------------------|
        | spatial_graph_{marker}.pkl | extracted_rois_{marker}.csv |
        | cell_features_{marker}.csv | (each row: ROI cell & attributes) |
        | best_spafgan_model_{marker}.pt | |
        In short:
        This step takes your spatial graph and cell features, uses the trained model to score and select the most interesting cells (ROIs), and outputs a CSV file with all relevant information for downstream analysis or visualization.



6-interaction_calculation.py

        Input Files:
        roi_cells_CD31.csv:
        Contains ROI information for CD31 marker
        Includes columns:
        roi_id: Unique ROI identifier
        cell_id: Cell identifier
        x, y, z: Spatial coordinates
        spafgan_score: SpaFGAN score
        Marker values (CD31, CD4, CD20, CD11b, CD11c, Catalase)
        roi_cells_CD11b.csv:
        Contains ROI information for CD11b marker
        Similar structure to CD31 file
        Output Files:
        roi_cells_CD31_interactions.csv:
        Same as input plus:
        Interaction scores for each ROI
        ROIs filtered with minimum 100 voxel distance
        Interaction columns:
        T-cell_entry_site: CD31 and CD4 interaction score
        B-cell_infiltration: CD20 and CD31 interaction score
        Oxidative_stress_niche: CD11b and Catalase interaction score
        Inflammatory_zone: CD11b and CD20 interaction score
        Dendritic_signal: CD11c interaction score
        roi_cells_CD11b_interactions.csv:
        Similar structure to CD31 file
        Focused on CD11b-related interactions
        Processing Steps:
        Read input files
        Calculate interaction scores for each ROI
        Filter ROIs based on minimum 100 voxel distance
        Save results to output files




7-input and output of generate_roi_shapes.py:
        Input Files:
        roi_cells_CD31_interactions.csv:
        Contains ROI data with interaction scores
        Columns include:
        roi_id: Unique identifier
        x, y, z: Spatial coordinates
        spafgan_score: SpaFGAN prediction score
        Interaction scores:
        T-cell_entry_site
        B-cell_infiltration
        Oxidative_stress_niche
        Inflammatory_zone
        Dendritic_signal
        Marker values (CD31, CD4, CD20, CD11b, etc.)
        roi_cells_CD11b_interactions.csv:
        Similar structure to CD31 file
        Contains ROI data specific to CD11b marker
        Output Files:
        roi_shapes.geojson:
        GeoJSON format file containing:
        Polygon shapes for each ROI
        Properties for each ROI:
        name: ROI identifier
        type: "ROI"
        score: SpaFGAN score
        interactions: List of detected interactions
        Spatial coordinates for visualization
        Used by the frontend for:
        Displaying ROI boundaries
        Showing interaction information
        Enabling interactive selection
        Processing Steps:
        Read the interaction CSV files
        For each ROI:
        Create a polygon shape using alpha shape algorithm
        Calculate centroid coordinates
        Add interaction properties
        Add marker values
        Combine all shapes into a single GeoJSON file
        Save the file for frontend visualization
        Would you like me to show you the code or run the script to generate these shapes?

8-explain the generate_vitessce_config_sdk.py script and its purpose:
        This script generates a configuration file for Vitessce, which is a visualization tool for single-cell and spatial data. Here's what the script does:
        Input Files:
        roi_shapes.geojson: Contains the cube shapes and properties of ROIs
        roi_cells_CD31_interactions.csv and roi_cells_CD11b_interactions.csv: Contain cell-level data with interaction scores
        Output:
        vitessce_config.json: A configuration file that tells Vitessce how to visualize:
        Spatial data (ROI shapes)
        Cell-level data
        Marker expressions
        Interaction scores
        Key Components:
        Spatial Layer: Shows ROI shapes as polygons
        Cell Layer: Displays individual cells with their properties
        Marker Layer: Visualizes marker expressions (CD31, CD11b, etc.)
        Interaction Layer: Shows interaction scores between cells




