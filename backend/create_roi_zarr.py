#!/usr/bin/env python3
"""
Create ROI Zarr file in the correct format for overlay on original image
"""

import zarr
import numpy as np
import json
from pathlib import Path

def create_roi_zarr_from_data(roi_data, output_path, zarr_name="roi_overlay.zarr"):
    """
    Create a Zarr file with ROI data in the correct format
    
    Parameters:
    -----------
    roi_data : dict
        Dictionary containing ROI data with keys:
        - 'x': list of X coordinates
        - 'y': list of Y coordinates  
        - 'radius': list of radii (optional, default=100)
        - 'roi_id': list of ROI IDs
        - 'interaction': list of ROI types
        - 'attention_score': list of attention scores (optional)
        - 'intensity_score': list of intensity scores (optional)
        - 'combined_score': list of combined scores (optional)
    
    output_path : str or Path
        Directory to save the Zarr file
    
    zarr_name : str
        Name of the Zarr file
    """
    
    # Ensure output directory exists
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    zarr_path = output_dir / zarr_name
    
    print(f"🔧 Creating ROI Zarr file: {zarr_path}")
    print("="*60)
    
    # Validate input data
    required_keys = ['x', 'y', 'roi_id', 'interaction']
    for key in required_keys:
        if key not in roi_data:
            raise ValueError(f"Missing required key: {key}")
    
    # Get number of ROIs
    n_rois = len(roi_data['x'])
    print(f"📊 Number of ROIs: {n_rois}")
    
    # Ensure all arrays have the same length
    for key, value in roi_data.items():
        if len(value) != n_rois:
            raise ValueError(f"Array '{key}' has length {len(value)}, expected {n_rois}")
    
    # Create Zarr store
    store = zarr.open_group(str(zarr_path), mode='w')
    
    # Set root attributes
    store.attrs['spatialdata_attrs'] = {
        'spatialdata_software_version': '0.4.0',
        'version': '0.1'
    }
    
    # Create shapes group
    shapes_group = store.create_group('shapes')
    roi_circles_group = shapes_group.create_group('roi_circles')
    
    # Set shapes attributes
    roi_circles_group.attrs['axes'] = ['x', 'y']
    roi_circles_group.attrs['coordinateTransformations'] = [
        {
            'input': {
                'axes': [
                    {'name': 'x', 'type': 'space', 'unit': 'unit'},
                    {'name': 'y', 'type': 'space', 'unit': 'unit'}
                ],
                'name': 'xy'
            },
            'output': {
                'axes': [
                    {'name': 'x', 'type': 'space', 'unit': 'unit'},
                    {'name': 'y', 'type': 'space', 'unit': 'unit'}
                ],
                'name': 'global'
            },
            'type': 'identity'
        }
    ]
    roi_circles_group.attrs['encoding-type'] = 'ngff:shapes'
    roi_circles_group.attrs['spatialdata_attrs'] = {'version': '0.2'}
    
    # Create arrays in shapes group
    print("📊 Creating shapes arrays:")
    
    # Required arrays
    roi_circles_group.create_dataset('x', data=np.array(roi_data['x'], dtype=np.int64), chunks=(n_rois,))
    print(f"  ✅ x: {roi_data['x'][:5]}...")
    
    roi_circles_group.create_dataset('y', data=np.array(roi_data['y'], dtype=np.int64), chunks=(n_rois,))
    print(f"  ✅ y: {roi_data['y'][:5]}...")
    
    # Optional radius (default to 100 if not provided)
    radius_data = roi_data.get('radius', [100] * n_rois)
    roi_circles_group.create_dataset('radius', data=np.array(radius_data, dtype=np.int32), chunks=(n_rois,))
    print(f"  ✅ radius: {radius_data[:5]}...")
    
    # ROI ID
    roi_id_data = roi_data['roi_id']
    roi_circles_group.create_dataset('roi_id', data=np.array(roi_id_data, dtype='U10'), chunks=(n_rois,))
    print(f"  ✅ roi_id: {roi_id_data[:5]}...")
    
    # Interaction type
    interaction_data = roi_data['interaction']
    roi_circles_group.create_dataset('interaction', data=np.array(interaction_data, dtype='U50'), chunks=(n_rois,))
    print(f"  ✅ interaction: {interaction_data[:5]}...")
    
    # Optional score arrays
    if 'attention_score' in roi_data:
        roi_circles_group.create_dataset('attention_score', data=np.array(roi_data['attention_score'], dtype=np.float64), chunks=(n_rois,))
        print(f"  ✅ attention_score: {roi_data['attention_score'][:5]}...")
    
    if 'intensity_score' in roi_data:
        roi_circles_group.create_dataset('intensity_score', data=np.array(roi_data['intensity_score'], dtype=np.float64), chunks=(n_rois,))
        print(f"  ✅ intensity_score: {roi_data['intensity_score'][:5]}...")
    
    if 'combined_score' in roi_data:
        roi_circles_group.create_dataset('combined_score', data=np.array(roi_data['combined_score'], dtype=np.float64), chunks=(n_rois,))
        print(f"  ✅ combined_score: {roi_data['combined_score'][:5]}...")
    
    # Create table group
    table_group = store.create_group('table')
    table_subgroup = table_group.create_group('table')
    
    print("\n📊 Creating table arrays:")
    
    # Create same arrays in table group (without radius)
    table_subgroup.create_dataset('x', data=np.array(roi_data['x'], dtype=np.int64), chunks=(n_rois,))
    table_subgroup.create_dataset('y', data=np.array(roi_data['y'], dtype=np.int64), chunks=(n_rois,))
    table_subgroup.create_dataset('roi_id', data=np.array(roi_id_data, dtype='U10'), chunks=(n_rois,))
    table_subgroup.create_dataset('interaction', data=np.array(interaction_data, dtype='U50'), chunks=(n_rois,))
    
    if 'attention_score' in roi_data:
        table_subgroup.create_dataset('attention_score', data=np.array(roi_data['attention_score'], dtype=np.float64), chunks=(n_rois,))
    if 'intensity_score' in roi_data:
        table_subgroup.create_dataset('intensity_score', data=np.array(roi_data['intensity_score'], dtype=np.float64), chunks=(n_rois,))
    if 'combined_score' in roi_data:
        table_subgroup.create_dataset('combined_score', data=np.array(roi_data['combined_score'], dtype=np.float64), chunks=(n_rois,))
    
    print("  ✅ Table arrays created")
    
    print(f"\n✅ ROI Zarr file created successfully: {zarr_path}")
    return str(zarr_path)

def create_sample_roi_data():
    """Create sample ROI data for testing"""
    return {
        'x': [6680, 56, 3456, 2552, 3400, 1200, 1800, 2400, 3000, 3600],
        'y': [3540, 1556, 2084, 1980, 2052, 1000, 1500, 2000, 2500, 3000],
        'radius': [100, 100, 100, 100, 100, 100, 100, 100, 100, 100],
        'roi_id': ['534', '369', '570', '427', '489', '600', '601', '602', '603', '604'],
        'interaction': ['B-cell infiltration', 'B-cell infiltration', 'T-cell entry site', 'Inflammatory zone', 'Oxidative stress niche', 
                       'B-cell infiltration', 'T-cell entry site', 'Inflammatory zone', 'Oxidative stress niche', 'B-cell infiltration'],
        'attention_score': [0.5, 0.4, 0.333, 0.429, 0.429, 0.6, 0.7, 0.8, 0.9, 0.3],
        'intensity_score': [0.282, 0.335, 0.382, 0.269, 0.263, 0.4, 0.5, 0.6, 0.7, 0.2],
        'combined_score': [0.391, 0.367, 0.357, 0.349, 0.346, 0.5, 0.6, 0.7, 0.8, 0.25]
    }

def main():
    """Main function to create sample ROI Zarr file"""
    print("🔧 ROI Zarr Creator")
    print("="*60)
    
    # Create sample data
    sample_data = create_sample_roi_data()
    
    # Create Zarr file
    output_path = Path(__file__).parent / "output"
    zarr_path = create_roi_zarr_from_data(sample_data, output_path, "sample_roi_overlay.zarr")
    
    print(f"\n📋 Summary:")
    print(f"  - Zarr file: {zarr_path}")
    print(f"  - ROIs: {len(sample_data['x'])}")
    print(f"  - ROI types: {set(sample_data['interaction'])}")
    print(f"  - Coordinate range: X({min(sample_data['x'])}-{max(sample_data['x'])}), Y({min(sample_data['y'])}-{max(sample_data['y'])})")
    
    print(f"\n🎯 Next steps:")
    print(f"1. Use this Zarr file in your Vitessce config")
    print(f"2. Replace sample data with your actual ROI data")
    print(f"3. Ensure coordinates match your original image")

if __name__ == "__main__":
    main() 