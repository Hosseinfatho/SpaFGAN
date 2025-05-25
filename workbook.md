# SpaFGAN Workbook

## Project Overview
This project is an interactive tool for spatial data visualization and analysis, utilizing React for the frontend and Python for the backend.

## Project Structure
- `frontend/`: React frontend code
  - `src/components/`: React components
    - `ROISelector.jsx`: ROI selection component with features:
      - Display and selection of interaction types
      - Navigation between ROIs
      - Manual coordinate adjustment
      - Display of scores and interaction information

## Workflow
1. Data Processing:
   - Load spatial data from Zarr format (level 3)
   - Process and normalize coordinates:
     * Scale X coordinates: factorX = 10908/1363
     * Scale Y coordinates: factorY = 5508/688
     * Flip Y coordinates for bottom-left origin
   - Extract ROI information:
     * Parse GeoJSON features
     * Calculate centroids for each ROI
     * Extract interaction types and scores
     * Handle both Polygon and MultiPolygon geometries

2. Backend Processing:
   - Serve ROI data through REST API:
     * Endpoint: `/api/roi_shapes`
     * Returns GeoJSON format with properties
   - Handle coordinate transformations:
     * Convert between Zarr levels
     * Apply scaling factors
     * Maintain coordinate system consistency
   - Manage interaction type filtering:
     * Group ROIs by interaction types
     * Support multiple selection
     * Enable dynamic filtering

3. Frontend Visualization:
   - Fetch and display ROI data:
     * Load data on component mount
     * Handle loading states
     * Display error messages if needed
   - Enable user interaction with ROIs:
     * Checkbox selection for interaction types
     * Previous/Next navigation
     * Manual coordinate input
   - Update view based on user selections:
     * Filter ROIs by selected interactions
     * Update current ROI index
     * Maintain selection state
   - Handle coordinate adjustments:
     * Validate manual inputs
     * Apply scaling factors
     * Update view parameters

4. User Interaction Flow:
   - Select interaction types of interest:
     * View available interaction types
     * Toggle selections
     * See immediate updates
   - Navigate through relevant ROIs:
     * Use Previous/Next buttons
     * View ROI details
     * Track current position
   - Adjust coordinates if needed:
     * Input X/Y values
     * See current coordinates
     * Reset to ROI center
   - Set view to focus on specific regions:
     * Apply coordinate changes
     * Update visualization
     * Maintain context
   - Analyze interaction patterns:
     * View interaction types
     * Check ROI scores
     * Compare different regions

## Usage Instructions
1. Run the backend server:
   ```bash
   cd backend
   python app.py
   ```

2. Run the frontend:
   ```bash
   cd frontend
   npm install
   npm start
   ```

3. Using ROISelector:
   - Select desired interaction types using checkboxes
   - Use Previous and Next buttons to navigate between ROIs
   - Manually adjust X and Y coordinates if needed
   - Use Set View button to apply changes

## Technical Notes
- Automatic coordinate scaling from Zarr level 3 to full resolution
- Support for multiple interaction types
- Display of ROI centroids
- Manual coordinate adjustment capability 