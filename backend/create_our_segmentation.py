import numpy as np
import tifffile
import json
import os

def create_ome_tiff_segmentation():
    """
    Create a segmentation file with 4 channels (interactions), size 5508x10908,
    read circle positions from JSON files, apply coordinate transformation, and save as ome-tiff.
    """
    # Parameters
    height, width = 5508, 10908
    channels = 4
    z_slices = 1
    time_points = 1
    radius = 200

    # Output array (T, C, Z, Y, X)
    data = np.zeros((time_points, channels, z_slices, height, width), dtype=np.uint8)

    # Interaction files and names
    interaction_files = [
        "output/top_roi_scores_B-cell_infiltration.json",
        "output/top_roi_scores_Inflammatory_zone.json",
        "output/top_roi_scores_Oxidative_stress_niche.json",
        "output/top_roi_scores_T-cell_entry_site.json"
    ]
    interaction_names = [
        "B-cell_infiltration",
        "Inflammatory_zone",
        "Oxidative_stress_niche",
        "T-cell_entry_site"
    ]

    for channel_idx, json_file in enumerate(interaction_files):
        if not os.path.exists(json_file):
            print(f"Warning: File {json_file} not found, skipping...")
            continue
        with open(json_file, 'r') as f:
            interaction_data = json.load(f)
        rois = interaction_data.get('top_rois', [])
        print(f"Channel {channel_idx} ({interaction_names[channel_idx]}): {len(rois)} ROIs")
        for roi in rois:
            pos = roi['position']
            x_orig = pos['x']
            y_orig = pos['y']
            x_final = int(round(x_orig * 8))
            y_final = int(round(height - y_orig * 8))
            # Draw circle
            y_grid, x_grid = np.ogrid[:height, :width]
            mask = (x_grid - x_final) ** 2 + (y_grid - y_final) ** 2 <= radius ** 2
            data[0, channel_idx, 0][mask] = 255
            print(f"  ROI {roi['roi_id']}: ({x_orig}, {y_orig}) -> ({x_final}, {y_final})")

    # Create simple OME-XML metadata
    ome_xml = f'''<?xml version="1.0" encoding="UTF-8"?>
<OME xmlns="http://www.openmicroscopy.org/Schemas/OME/2016-06">
  <Image ID="Image:0" Name="obsSegmentations">
    <Pixels ID="Pixels:0" Type="uint8" DimensionOrder="TCZYX" BigEndian="false"
      SizeX="{width}" SizeY="{height}" SizeZ="{z_slices}" SizeC="{channels}" SizeT="{time_points}">
      {''.join([f'<Channel ID="Channel:0:{i}" Name="{interaction_names[i]}" SamplesPerPixel="1"/>' for i in range(channels)])}
    </Pixels>
  </Image>
</OME>'''

    output_file = "output/vitessce_files/obsSegmentations.ome.tiff"
    os.makedirs("output", exist_ok=True)
    tifffile.imwrite(
        output_file,
        data,
        photometric='minisblack',
        metadata={'axes': 'TCZYX', 'Description': ome_xml},
        compression='lzw'
    )
    print(f"\nCreated segmentation file: {output_file}")
    print(f"Shape: {data.shape}")
    print(f"Data type: {data.dtype}")
    print(f"Unique values: {np.unique(data)}")

if __name__ == "__main__":
    create_ome_tiff_segmentation() 