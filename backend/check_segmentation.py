import tifffile
import numpy as np

# Load the segmentation file
tiff = tifffile.TiffFile('output/vitessce_files/obsSegmentations.ome.tiff')

print("=== Segmentation File Analysis ===")
print(f"Shape: {tiff.series[0].shape}")
print(f"Dtype: {tiff.series[0].dtype}")

# Check if it's multi-channel
if len(tiff.series[0].shape) > 2:
    num_channels = tiff.series[0].shape[0]
    print(f"Number of channels: {num_channels}")
    
    # Check unique values in each channel
    for i in range(min(num_channels, 4)):  # Check first 4 channels
        channel_data = tiff.series[0].asarray()[i]
        unique_vals = np.unique(channel_data)
        print(f"Channel {i} unique values: {unique_vals}")
        print(f"Channel {i} value range: {channel_data.min()} to {channel_data.max()}")
else:
    print("Single channel image")
    data = tiff.series[0].asarray()
    unique_vals = np.unique(data)
    print(f"Unique values: {unique_vals}")
    print(f"Value range: {data.min()} to {data.max()}")

print("=== End Analysis ===") 