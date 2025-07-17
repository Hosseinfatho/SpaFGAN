import tifffile
import os

def analyze_tiff_metadata():
    tiff_path = os.path.abspath("output/vitessce_files/S-1905-017737_PAS_2of2.ome.tif")
    print("=== TIFF File Metadata Analysis ===")
    print(f"File path: {tiff_path}")
    print(f"File exists: {os.path.exists(tiff_path)}")
    print()
    with tifffile.TiffFile(tiff_path) as tif:
        print(f"Number of pages: {len(tif.pages)}")
        print(f"Image shape (first page): {tif.pages[0].shape}")
        print(f"Dtype (first page): {tif.pages[0].dtype}")
        print(f"Description: {tif.pages[0].description}")
        print(f"Tags (first page):")
        for tag in tif.pages[0].tags.values():
            print(f"  {tag.name}: {tag.value}")
        print("\nOME-XML (if present):")
        if tif.ome_metadata:
            print(tif.ome_metadata)
        else:
            print("No OME-XML metadata found.")

if __name__ == "__main__":
    analyze_tiff_metadata() 