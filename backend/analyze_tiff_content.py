import tifffile
import numpy as np
from pathlib import Path

def analyze_tiff_content():
    """Analyze the actual content of the TIFF file to find segmentation IDs"""
    
    tiff_file = Path("output/vitessce_files/S-1905-017737_PAS_2of2.ome.tif")
    
    print(f"Analyzing TIFF content: {tiff_file}")
    print("=" * 80)
    
    with tifffile.TiffFile(tiff_file) as tif:
        print(f"Number of pages: {len(tif.pages)}")
        print(f"Series shape: {tif.series[0].shape}")
        
        # Read only a small portion to avoid memory issues
        # Read first 1000x1000 pixels from each channel
        sample_size = 1000
        
        for i in range(min(7, len(tif.pages))):
            print(f"\n--- Channel {i} ---")
            
            # Read a small sample
            try:
                # Read first page directly
                page_data = tif.pages[i].asarray()
                print(f"Page {i} shape: {page_data.shape}")
                
                # Take a sample from the top-left corner
                sample = page_data[:sample_size, :sample_size]
                print(f"Sample shape: {sample.shape}")
                
                # Find unique values in sample
                unique_vals = np.unique(sample)
                print(f"Unique values in sample: {unique_vals}")
                print(f"Value range: {sample.min()} to {sample.max()}")
                
                # Count occurrences of each value in sample
                for val in unique_vals:
                    if val > 0:  # Skip background (0)
                        count = np.sum(sample == val)
                        print(f"  ID {val}: {count} pixels in sample")
                
                # Look for specific ID 9
                if 9 in unique_vals:
                    print(f"  *** Found ID 9 in channel {i} sample ***")
                    coords = np.where(sample == 9)
                    if len(coords[0]) > 0:
                        print(f"    Sample coordinates: ({coords[0][0]}, {coords[1][0]}) to ({coords[0][-1]}, {coords[1][-1]})")
                        print(f"    Total pixels in sample: {len(coords[0])}")
                
            except Exception as e:
                print(f"Error reading channel {i}: {e}")

if __name__ == "__main__":
    analyze_tiff_content() 