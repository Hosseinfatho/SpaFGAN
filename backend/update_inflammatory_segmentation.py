import json
from pathlib import Path

def update_inflammatory_segmentation():
    """Update the inflammatory zone segmentation file to include all ROIs from top ROI scores"""
    
    # Load the top ROI scores file
    top_roi_file = Path(__file__).parent / 'output' / 'top_roi_scores_Inflammatory zone.json'
    with open(top_roi_file, 'r') as f:
        top_roi_data = json.load(f)
    
    # Load the rectangles annotation file (contains all ROI geometries)
    rectangles_file = Path(__file__).parent / 'output' / 'roi_rectangles_annotation.json'
    with open(rectangles_file, 'r') as f:
        rectangles_data = json.load(f)
    
    # Create new segmentation data
    new_segmentation = {}
    
    # For each ROI in top ROI scores, get its geometry from rectangles
    for roi in top_roi_data['top_rois']:
        roi_id = roi['roi_id']
        tooltip_name = roi['tooltip_name']
        
        # Create the key format that matches the segmentation file
        # Extract the score from tooltip_name
        score_part = tooltip_name.split('Score:')[1]
        key = f"Inflammatory zone_{roi_id}_Score:{score_part}"
        
        # Look for this ROI in the rectangles data
        # Try different possible key formats
        possible_keys = [
            key,
            tooltip_name,
            f"Inflammatory zone_{roi_id}_Score:{roi['scores']['combined_score']:.3f}",
            f"Inflammatory zone_{roi_id}"
        ]
        
        found_geometry = None
        for possible_key in possible_keys:
            if possible_key in rectangles_data:
                found_geometry = rectangles_data[possible_key]
                break
        
        if found_geometry:
            new_segmentation[key] = found_geometry
            print(f"Added ROI {roi_id} with key: {key}")
        else:
            print(f"Warning: Could not find geometry for ROI {roi_id}")
    
    # Save the updated segmentation file
    output_file = Path(__file__).parent / 'output' / 'roi_segmentation_Inflammatory_zone.json'
    with open(output_file, 'w') as f:
        json.dump(new_segmentation, f, indent=2)
    
    print(f"Updated segmentation file with {len(new_segmentation)} ROIs")

if __name__ == "__main__":
    update_inflammatory_segmentation() 