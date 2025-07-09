import zarr
import numpy as np
import json
from pathlib import Path

def create_simple_segmentation_zarr():
    """Create a simple segmentation.ome-zarr file for testing"""
    
    # Create output directory
    output_dir = Path("output/segmentation.ome-zarr")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create a simple segmentation mask (100x100 pixels)
    # This will be a simple rectangular ROI
    segmentation_mask = np.zeros((100, 100), dtype=np.uint32)
    
    # Create a simple rectangular ROI (20x30 pixels starting at (10, 10))
    segmentation_mask[10:40, 10:30] = 1  # ROI ID = 1
    
    # Create another ROI
    segmentation_mask[50:80, 60:90] = 2  # ROI ID = 2
    
    # Create zarr store
    store = zarr.open(str(output_dir), mode='w')
    
    # Create the segmentation array
    segmentation_array = store.create_dataset(
        '0/labels/segmentation',
        data=segmentation_mask,
        chunks=(50, 50),
        dtype=np.uint32
    )
    
    # Add OME metadata
    ome_metadata = {
        "omero": {
            "id": 1,
            "name": "segmentation",
            "version": "0.4",
            "channels": [
                {
                    "id": 0,
                    "label": "segmentation",
                    "color": "00ff00",
                    "window": {"start": 0, "end": 255},
                    "active": True
                }
            ]
        },
        "multiscales": [
            {
                "version": "0.4",
                "axes": [
                    {"name": "y", "type": "space", "unit": "pixel"},
                    {"name": "x", "type": "space", "unit": "pixel"}
                ],
                "datasets": [
                    {
                        "path": "0/labels/segmentation",
                        "coordinateTransformations": [
                            {
                                "type": "scale",
                                "scale": [1.0, 1.0]
                            }
                        ]
                    }
                ]
            }
        ]
    }
    
    # Save metadata
    with open(output_dir / "0" / "labels" / "segmentation" / ".zattrs", 'w') as f:
        json.dump(ome_metadata, f, indent=2)
    
    # Create obsSegmentations.json for Vitessce
    obs_segmentations = {
        "obsIndex": ["roi_1", "roi_2"],
        "obsSegmentations": [
            {
                "name": "roi_1",
                "type": "polygon",
                "coordinates": [[10, 10], [30, 10], [30, 40], [10, 40], [10, 10]]
            },
            {
                "name": "roi_2", 
                "type": "polygon",
                "coordinates": [[60, 50], [90, 50], [90, 80], [60, 80], [60, 50]]
            }
        ]
    }
    
    with open(output_dir / "obsSegmentations.json", 'w') as f:
        json.dump(obs_segmentations, f, indent=2)
    
    print(f"Created segmentation.ome-zarr at {output_dir}")
    return str(output_dir)

if __name__ == "__main__":
    create_simple_segmentation_zarr() 