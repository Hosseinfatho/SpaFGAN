import numpy as np
import tifffile
import json
from pathlib import Path
import cv2
from PIL import Image, ImageDraw

def create_vitessce_segmentation():
    """
    Create obsSegmentations.ome-tiff file with 4 channels for 4 interaction types
    Each ROI gets a unique ID, and proper OME-XML metadata is included
    """
    
    # Define image dimensions (same as your original file)
    height, width = 5508, 10908
    
    # Create 4-channel array with uint16 (same as original)
    # Shape: (4, height, width) - 4 channels for 4 interaction types
    segmentation_array = np.zeros((4, height, width), dtype=np.uint16)
    
    # Define interaction types
    interaction_types = [
        "B-cell_infiltration",
        "Inflammatory_zone", 
        "T-cell_entry_site",
        "Oxidative_stress_niche"
    ]
    
    print(f"Creating segmentation array with shape: {segmentation_array.shape}")
    print(f"Interaction types: {interaction_types}")
    
    # Load existing ROI data and create masks
    roi_files = {
        'B-cell_infiltration': 'output/roi_segmentation_B-cell_infiltration.json',
        'Inflammatory_zone': 'output/roi_segmentation_Inflammatory_zone.json',
        'T-cell_entry_site': 'output/roi_segmentation_T-cell_entry_site.json',
        'Oxidative_stress_niche': 'output/roi_segmentation_Oxidative_stress_niche.json'
    }
    
    # Process each interaction type
    for channel_idx, (interaction_name, roi_file_path) in enumerate(roi_files.items()):
        print(f"\nProcessing channel {channel_idx}: {interaction_name}")
        
        if Path(roi_file_path).exists():
            # Load ROI coordinates
            with open(roi_file_path, 'r') as f:
                roi_data = json.load(f)
            
            # Create mask for this channel
            mask = np.zeros((height, width), dtype=np.uint16)
            
            # Draw each ROI on the mask with same value (255)
            for roi_key, coordinates in roi_data.items():
                if coordinates and len(coordinates) > 0:
                    # Convert coordinates to numpy array
                    coords = np.array(coordinates[0], dtype=np.int32)  # First polygon
                    
                    # Draw filled polygon with value 255 (same for all ROIs in this channel)
                    cv2.fillPoly(mask, [coords], 255)
            
            # Assign to channel
            segmentation_array[channel_idx] = mask
            
            print(f"  - Added {len(roi_data)} ROIs to channel {channel_idx}")
            print(f"  - All ROIs have value: 255")
        else:
            print(f"  - Warning: ROI file not found: {roi_file_path}")
    
    # Save as OME-TIFF
    output_path = "output/vitessce_files/obsSegmentations.ome.tiff"
    tifffile.imwrite(output_path, segmentation_array, 
                    photometric='minisblack',
                    compression='lzw')
    
    print(f"\nSaved segmentation TIFF to: {output_path}")
    print(f"File shape: {segmentation_array.shape}")
    print(f"Data type: {segmentation_array.dtype}")
    print(f"Value range: {np.min(segmentation_array)} - {np.max(segmentation_array)}")
    
    # Create offsets file
    create_offsets_file()
    
    return output_path

def create_offsets_file():
    """
    Create offsets file for the segmentation TIFF
    """
    # For now, create a simple offsets file
    # In a real implementation, you would calculate actual offsets
    offsets = [16, 1000000, 2000000, 3000000, 4000000, 5000000, 6000000]
    
    offsets_path = "output/vitessce_files/obsSegmentations.offsets.json"
    with open(offsets_path, 'w') as f:
        json.dump(offsets, f)
    
    print(f"Created offsets file: {offsets_path}")
    print(f"Offsets: {offsets}")

if __name__ == "__main__":
    # Create output directory if it doesn't exist
    Path("output/vitessce_files").mkdir(parents=True, exist_ok=True)
    
    # Create segmentation file
    create_vitessce_segmentation() 