# backend/test_zarr_exploration.py

import numpy as np
import time

# --- Import functions from server.py ---
# Note: For this to work, the backend directory must be in PYTHONPATH
# or you run this script from the workspace root (e.g., python backend/test_zarr_exploration.py)
# Alternatively, adjust sys.path if needed.
from server import (
    open_target_zarr_array, # To get the zarr array
    calculate_histogram,
    calculate_z_projection_heatmap,
    print_channel_names # If you still need to test this separately
)

# --- Main Test Execution ---
if __name__ == "__main__":
    print("--- Running Analysis Function Tests via server.py functions ---")

    # 1. Open the Zarr array using the server's helper function
    print("Attempting to open Zarr array via server.open_target_zarr_array()...")
    target_image_arr = open_target_zarr_array()

    if target_image_arr is not None:
        print(f"Successfully opened target Zarr array. Shape: {target_image_arr.shape}")
        
        # (Optional) Test print_channel_names if needed, requires group objects
        # You might need to adapt how you get image_group and root_group for this test
        # print("\\n--- Testing print_channel_names (SKIPPED - requires group objects) ---")

        # --- Define Test Parameters (Ensure they are valid for the array shape) ---
        # Example shape: (1, 70, 194, 1377, 2727) for t,c,z,y,x
        test_channel_index = 25 
        # ROI for RESOLUTION 2 (as opened by open_target_zarr_array)
        test_roi_z = (10, 110)      # Within Z=194
        test_roi_y_bottom_up = (100, 300) # Y is bottom-up for heatmap, within Y_max=1377
        test_roi_x = (1200, 1400)   # Within X=2727
        test_num_bins = 50
        test_projection_type = 'mean'

        print(f"\\nTest Params: Channel={test_channel_index}, Bins={test_num_bins}")
        print(f"  ROI (bottom-up for heatmap): Z={test_roi_z}, Y={test_roi_y_bottom_up}, X={test_roi_x}")
        print(f"  Projection Type for Heatmap: {test_projection_type}")

        # --- Test calculate_histogram --- 
        # Histogram expects Y as is (top-down if zarr is top-down for its direct slicing)
        # For consistency with heatmap, if you want to test histogram with bottom-up Y, you'd need to transform it here too.
        # However, the current calculate_histogram in server.py doesn't do Y transformation.
        # Let's use a direct Zarr-compatible Y for histogram test for now.
        test_roi_y_hist_zarr = (600, 800) # Example direct slice for Zarr
        print(f"  ROI (for histogram - direct Zarr slice): Z={test_roi_z}, Y={test_roi_y_hist_zarr}, X={test_roi_x}")

        print("\\n--- Testing calculate_histogram (from server.py) --- ")
        hist_result = calculate_histogram(
            target_image_arr,
            test_channel_index,
            test_roi_z,
            test_roi_y_hist_zarr, # Using direct Zarr Y for histogram
            test_roi_x,
            test_num_bins
        )
        if 'error' in hist_result:
            print(f"Histogram Test FAILED: {hist_result['error']}")
        else:
            print("Histogram Test SUCCEEDED.")
            # print(f"  Counts: {hist_result['counts'][:5]}...")
            # print(f"  Bin Edges: {hist_result['bin_edges'][:6]}...")

        # --- Test calculate_z_projection_heatmap ---
        print("\\n--- Testing calculate_z_projection_heatmap (from server.py) --- ")
        heatmap_result = calculate_z_projection_heatmap(
            target_image_arr,
            test_channel_index,
            test_roi_z,
            test_roi_y_bottom_up, # Using bottom-up Y for heatmap function
            test_roi_x,
            test_projection_type
        )
        if 'error' in heatmap_result:
            print(f"Heatmap Test FAILED: {heatmap_result['error']}")
        else:
            print("Heatmap Test SUCCEEDED.")
            # print(f"  Heatmap Shape: {heatmap_result['shape']}")
            # if heatmap_result['heatmap'] and heatmap_result['heatmap'][0]:
            #    print(f"  Heatmap Top-Left 5x5: {np.array(heatmap_result['heatmap'])[:5,:5]}")

    else:
        print("\\n--- Skipping Analysis Function Tests (Target Zarr array not loaded via server.py) ---")

    print("\\n--- End of Test Script ---")