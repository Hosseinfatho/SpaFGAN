import tifffile
from pathlib import Path

def read_ome_metadata():
    """Read and display the exact OME metadata from the original TIFF file"""
    
    tiff_file = Path("output/vitessce_files/S-1905-017737_PAS_2of2.ome.tif")
    
    print(f"Reading OME metadata from: {tiff_file}")
    print("=" * 80)
    
    with tifffile.TiffFile(tiff_file) as tif:
        # Get OME metadata from the first page
        first_page = tif.pages[0]
        
        print("OME metadata from ImageDescription:")
        print("-" * 40)
        
        if hasattr(first_page, 'description') and first_page.description:
            ome_metadata = first_page.description
            print(ome_metadata)
        else:
            print("No description found in first page")
        
        print("\n" + "=" * 80)
        print("OME metadata from tif.ome_metadata:")
        print("-" * 40)
        
        if hasattr(tif, 'ome_metadata') and tif.ome_metadata:
            print(tif.ome_metadata)
        else:
            print("No ome_metadata found")
        
        print("\n" + "=" * 80)
        print("All pages descriptions:")
        print("-" * 40)
        
        for i, page in enumerate(tif.pages):
            if hasattr(page, 'description') and page.description:
                print(f"\nPage {i} description:")
                print(page.description)
            else:
                print(f"\nPage {i}: No description")

if __name__ == "__main__":
    read_ome_metadata() 