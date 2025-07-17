import tifffile
import numpy as np
import json
from pathlib import Path

def analyze_tiff_structure(file_path):
    """Analyze the structure of the original TIFF file without loading full image"""
    
    print(f"Analyzing: {file_path}")
    
    try:
        with tifffile.TiffFile(file_path) as tif:
            print(f"Number of pages: {len(tif.pages)}")
            
            # Check each page individually without loading full data
            for i, page in enumerate(tif.pages):
                print(f"\nPage {i}:")
                print(f"  Shape: {page.shape}")
                print(f"  Dtype: {page.dtype}")
                print(f"  Compression: {page.compression}")
                print(f"  Photometric: {page.photometric}")
                print(f"  Planar config: {page.planarconfig}")
                
                # Try to read a small sample
                try:
                    # Read just a small region to understand the data
                    sample = page.asarray(key=slice(0, min(100, page.shape[0])), 
                                        col=slice(0, min(100, page.shape[1])))
                    unique_values = np.unique(sample)
                    print(f"  Sample unique values: {unique_values[:10]}... (total: {len(unique_values)})")
                    print(f"  Sample min: {sample.min()}, max: {sample.max()}")
                except Exception as e:
                    print(f"  Could not read sample: {e}")
            
            # Check OME metadata
            if hasattr(tif, 'ome_metadata'):
                print(f"\nOME metadata available: {tif.ome_metadata is not None}")
                if tif.ome_metadata:
                    print("OME metadata content:")
                    print(tif.ome_metadata[:1000] + "..." if len(tif.ome_metadata) > 1000 else tif.ome_metadata)
            
            # Check image description
            for i, page in enumerate(tif.pages):
                if hasattr(page, 'description') and page.description:
                    print(f"\nPage {i} description: {page.description[:200]}...")
    except Exception as e:
        print(f"Error reading TIFF file: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Analyze the original file
    original_file = Path("output/vitessce_files/S-1905-017737_PAS_2of2.ome.tif")
    
    if original_file.exists():
        analyze_tiff_structure(original_file)
    else:
        print(f"File not found: {original_file}") 