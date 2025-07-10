import s3fs
import zarr

# AWS OME-Zarr path
ZARR_BASE_URL = "s3://lsp-public-data/yapp-2023-3d-melanoma/Dataset1-LSP13626-melanoma-in-situ"
s3 = s3fs.S3FileSystem(anon=True)
store = s3fs.S3Map(root=ZARR_BASE_URL, s3=s3, check=False)
root = zarr.open_consolidated(store) if '.zmetadata' in store else zarr.open_group(store, mode='r')

# Recursively print the group/array tree structure
# This helps to understand the exact structure for dummy data creation

def print_tree(g, prefix=""):
    print(prefix + str(g))
    for k, v in g.groups():
        print_tree(v, prefix + "  ")
    for k, v in g.arrays():
        print(prefix + "  " + str(v))

print("=== OME-Zarr Group/Array Structure ===")
print_tree(root)

# Print attributes and shape of main arrays
print("\n=== Attributes and Shapes ===")
print("Root attrs:", root.attrs.asdict())
if "0" in root:
    g0 = root["0"]
    print("\nGroup '0' attrs:", g0.attrs.asdict())
    for k in g0:
        print(f"\nSubgroup '{k}' attrs:", g0[k].attrs.asdict())
        if hasattr(g0[k], "shape"):
            print(f"Shape: {g0[k].shape}, dtype: {g0[k].dtype}, chunks: {g0[k].chunks}") 