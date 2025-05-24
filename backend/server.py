# Backend server code will go here 
import json
import logging
from flask import Flask, jsonify, request, Response, stream_with_context
from flask_cors import CORS
import requests
from vitessce import (
    VitessceConfig,
    CoordinationLevel as CL,
    get_initial_coordination_scope_prefix,
    Component
)
# Removed unused ipywidgets import
import zarr as pyzarr # Use alias to avoid confusion with js zarr
import numpy as np
from scipy import ndimage
from skimage.measure import label # Used for connected components if needed, optional
import traceback # For better error logging
import s3fs # Needed for opening S3 Zarr store
import time # For timing operations, used in copied functions
import base64
import io
from PIL import Image # Pillow import
import os
import pickle
from pathlib import Path

# --- Analysis Functions (Copied from test_zarr_exploration.py) ---
# Configure basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
# Enable CORS for requests from the frontend development server
CORS(app, resources={r"/api/*": {"origins": "http://localhost:5173"}}) # Adjust origin port if Vite uses a different one

# --- Constants for Zarr Access ---
ZARR_BASE_URL = "s3://lsp-public-data/yapp-2023-3d-melanoma/Dataset1-LSP13626-melanoma-in-situ"
ZARR_IMAGE_GROUP_PATH = "0"
TARGET_RESOLUTION_PATH = "3" # Use the target resolution
# ---
REQUIRED_CHANNEL_INDICES_FOR_RGB_INTERACTION = {
    'CD31': 19,
    'CD11b': 37,
    'Catalase': 59,
    'CD4': 25,
    'CD20': 27,
    'CD11c': 41
}
# Define the interaction compositions and their target RGB channels
RGB_INTERACTION_DEFINITIONS = [
    {'name': 'Vascular inflammation + oxidative stress', 'channels': ['CD31', 'CD11b', 'Catalase'], 'rgb_target': 0}, # Red
    {'name': 'Complex immune cell interaction', 'channels': ['CD4', 'CD20', 'CD11b'], 'rgb_target': 2}, # Blue
    {'name': 'T/B cell entry via vessels', 'channels': ['CD31', 'CD4', 'CD20'], 'rgb_target': 1}  # Green
]
# --- New Endpoint for Channel Names ---
@app.route('/api/channel_names')
def get_channel_names():
    """Reads OME-Zarr metadata to get channel names."""
    logger.info("Request received for /api/channel_names")
    channel_map = {}
    s3_local = None
    root_store = None # Initialize here for broader scope in logging
    root_group = None # Initialize here
    try:
        logger.info("Attempting to create S3FileSystem...")
        s3_local = s3fs.S3FileSystem(anon=True)
        logger.info("S3FileSystem created successfully.")
        
        logger.info(f"Attempting to create S3Map for root: {ZARR_BASE_URL}")
        root_store = s3fs.S3Map(root=ZARR_BASE_URL, s3=s3_local, check=False)
        logger.info("S3Map created successfully.")

        logger.info("Attempting to open root Zarr group...")
        # Open the root group, not a specific resolution array
        root_group = pyzarr.open_consolidated(store=root_store) if '.zmetadata' in root_store else pyzarr.open_group(store=root_store, mode='r')
        logger.info(f"Root group opened successfully. Type: {type(root_group)}")

        # Access OME metadata: Look inside the image group (e.g., '0')
        omero_meta = None
        image_group_key = '0' # Assuming the main image data is in group '0'
        logger.info(f"Checking for image group '{image_group_key}'...")
        if root_group and image_group_key in root_group:
            image_group = root_group[image_group_key]
            logger.info(f"Image group '{image_group_key}' found. Checking for 'omero' in its attributes...")
            if hasattr(image_group, 'attrs') and 'omero' in image_group.attrs:
                omero_meta = image_group.attrs['omero']
                logger.info(f"Found 'omero' metadata in image group '{image_group_key}' attributes.")
            else:
                 logger.warning(f"Could not find 'omero' metadata in image group '{image_group_key}' attributes.")
                 # Optional: Log available attributes for debugging
                 # if hasattr(image_group, 'attrs'):
                 #    logger.debug(f"Available image group attributes: {list(image_group.attrs.keys())}")
        else:
             logger.warning(f"Image group '{image_group_key}' not found in root group.")

        # Now check the extracted omero_meta for channels
        if omero_meta and 'channels' in omero_meta:
            logger.info(f"Found {len(omero_meta['channels'])} channels in metadata.")
            for i, channel_info in enumerate(omero_meta['channels']): 
                # Ensure key is string for JSON compatibility
                channel_map[str(i)] = channel_info.get('label', f'Channel {i}') 
            logger.info(f"✅ Successfully extracted channel names: {channel_map}")
            return jsonify(channel_map)
        else:
            logger.warning("Could not find valid 'omero' metadata with 'channels'. Returning 404.")
            # Return empty map or error?
            return jsonify({"error": "Channel metadata ('omero' with 'channels') not found in Zarr store"}), 404

    except Exception as e:
        logger.error(f"❌ Failed during channel name retrieval: {e}", exc_info=True)
        # Log details about variables at the time of error
        logger.error(f"State at error: s3_local={'Exists' if s3_local else 'None'}, root_store={'Exists' if root_store else 'None'}, root_group={'Exists' if root_group else 'None'}")
        return jsonify({"error": f"Failed to read channel names: {e}"}), 500
# --- End Channel Names Endpoint ---
# --- RGB Interaction Heatmap Function ---
# Define the fixed channel indices based on the provided mapping
# This map should ideally be managed more robustly if channel names/indices can change
# For now, hardcoding based on user's provided list.
# CD31: 19, CD11b: 37, Catalase: 59, CD4: 25, CD20: 27, Cd11c:41

# Function to extract and print channel names from OME metadata, and explore attributes
def print_channel_names(image_group, root_group=None, target_array=None):
    """
    Attempts to read OME metadata and prints channel names.
    Also prints all available attributes for the provided groups/array to help find metadata.

    Args:
        image_group (zarr.Group): The main image group (e.g., /0).
        root_group (zarr.Group, optional): The root Zarr group.
        target_array (zarr.Array, optional): The specific resolution array (e.g., /0/2).
    """
    # Using logger instead of print for server-side logging
    logger.info("\\n--- Exploring Metadata Attributes ---")

    groups_to_check = {
        "Image Group (/0)": image_group,
        "Root Group (/) ": root_group,
        "Target Array (/0/3)": target_array
    }

    found_channels = False

    for name, item in groups_to_check.items():
        if item is None:
            logger.info(f"\\n{name}: Not provided or not loaded.")
            continue

        logger.info(f"\\n{name}:")
        try:
            if hasattr(item, 'attrs'):
                logger.info(f"  Available attributes: {list(item.attrs.keys())}")

                # Try the standard OME check
                if 'omero' in item.attrs:
                    omero_meta = item.attrs['omero']
                    if isinstance(omero_meta, dict) and 'channels' in omero_meta:
                        channels = omero_meta['channels']
                        if isinstance(channels, list):
                            logger.info(f"  Found {len(channels)} channels via OME metadata in {name}:")
                            for i, channel_info in enumerate(channels):
                                if isinstance(channel_info, dict):
                                    label = channel_info.get('label', f'Channel {i}')
                                    logger.info(f"    Index {i}: {label}")
                                else:
                                     logger.info(f"    Index {i}: Malformed channel metadata entry")
                            logger.info("  --------------------------------------")
                            found_channels = True # Mark as found
                        else:
                            logger.info("  [Info] 'omero':{'channels'} found, but value is not a list.")
                    else:
                         logger.info("  [Info] 'omero' attribute found, but no 'channels' key inside or not a dict.")
                else:
                    logger.info("  [Info] Standard 'omero' key not found in attributes.")
            else:
                 logger.info("  [Info] This item has no attributes.")
        except Exception as e:
            logger.error(f"  [Error] An error occurred while inspecting attributes for {name}: {e}")

    if not found_channels:
        logger.info("\\n--- Summary: Standard OME channel names not found in checked locations. ---")
        logger.info("Review the printed attributes above to see if names are stored under different keys.")
    logger.info("--------------------------------------")

    """Calculates the histogram for a specific channel within a defined ROI."""
    if zarr_array is None:
        return {'error': 'Zarr array is None.'}
    try:
        if zarr_array.ndim != 5:
            logger.warning(f"[Histogram] Expected 5D array (t,c,z,y,x), but got {zarr_array.ndim}D. Proceeding cautiously.")
        channel_axis = 1
        if not (0 <= channel_index < zarr_array.shape[channel_axis]):
            return {'error': f'Invalid channel_index {channel_index}. Must be between 0 and {zarr_array.shape[channel_axis] - 1}. Array shape: {zarr_array.shape}'}
        max_z_arr, max_y_arr, max_x_arr = zarr_array.shape[2:]
        if not (0 <= roi_z[0] < roi_z[1] <= max_z_arr and
                0 <= roi_y[0] < roi_y[1] <= max_y_arr and
                0 <= roi_x[0] < roi_x[1] <= max_x_arr):
            return {'error': f'ROI Z{roi_z}, Y{roi_y}, X{roi_x} is out of bounds for array shape Z{max_z_arr}, Y{max_y_arr}, X{max_x_arr}.'}
        
        total_y_height_in_zarr = zarr_array.shape[3]
        zarr_slice_y_start = total_y_height_in_zarr - roi_y[1]
        zarr_slice_y_end = total_y_height_in_zarr - roi_y[0]
        zarr_slice_y_start = max(0, zarr_slice_y_start)
        zarr_slice_y_end = min(total_y_height_in_zarr, zarr_slice_y_end)

        if zarr_slice_y_start >= zarr_slice_y_end and not (roi_y[0] == 0 and roi_y[1] == total_y_height_in_zarr):
            return {'error': f'Transformed Y ROI resulted in invalid slice: [{zarr_slice_y_start}, {zarr_slice_y_end}). Original roi_y: {roi_y}, Zarr Y height: {total_y_height_in_zarr}'}
        
        logger.info(f"[Histogram] Original roi_y (bottom-up): {roi_y}")
        logger.info(f"[Histogram] Transformed Zarr Y slice (top-down): [{zarr_slice_y_start}, {zarr_slice_y_end})")
        
        roi_slice = (
            slice(0, 1),
            slice(channel_index, channel_index + 1),
            slice(roi_z[0], roi_z[1]),
            slice(zarr_slice_y_start, zarr_slice_y_end),
            slice(roi_x[0], roi_x[1])
        )
        logger.info(f"[Histogram] Reading slice: Ch={channel_index}, Z_roi={roi_z}, Y_ZARR_SLICE=[{zarr_slice_y_start}:{zarr_slice_y_end}], X_roi={roi_x} from array shape {zarr_array.shape}")
        start_read = time.time()
        data_roi = zarr_array[roi_slice]
        end_read = time.time()
        logger.info(f"[Histogram] Read in {end_read - start_read:.2f}s. Result shape: {data_roi.shape}, Size: {data_roi.size}")
        if data_roi.size == 0:
            return {'error': 'Specified ROI resulted in an empty data slice.'}

        # Flatten the data and convert to float32
        data_flat = data_roi.astype(np.float32).flatten()
        
        # Get min and max values
        min_val = np.min(data_flat)
        max_val = np.max(data_flat)
        
        # Create bins based on actual data range
        bins = np.linspace(min_val, max_val, num_bins + 1)
        
        # Calculate histogram
        counts, bin_edges = np.histogram(data_flat, bins=bins)
        
        logger.info(f"[Histogram] Calculated histogram with {num_bins} bins. Range: [{min_val:.2f}, {max_val:.2f}]")
        logger.info(f"[Histogram] Non-zero bins: {np.count_nonzero(counts)}")
        
        return {
            'counts': counts.tolist(),
            'bin_edges': bin_edges.tolist()
        }
    except IndexError as ie:
        logger.error(f"[Histogram] IndexError: {ie}", exc_info=True)
        return {'error': f'ROI coordinates likely out of bounds. Details: {ie}'}
    except ValueError as ve:
        logger.error(f"[Histogram] ValueError: {ve}", exc_info=True)
        return {'error': f'Invalid ROI values? ROI: Z{roi_z}, Y{roi_y}, X{roi_x}. Details: {ve}'}
    except Exception as e:
        logger.error(f"[Histogram] Unexpected Error: {e}", exc_info=True)
        return {'error': f'Unexpected error during histogram calculation: {e}'}

    """Calculates the Z-projection (heatmap) for a specific channel within a defined ROI."""
    if zarr_array is None:
        return {'error': 'Zarr array is None.'}
    valid_projection_types = ['mean', 'max']
    if projection_type not in valid_projection_types:
        return {'error': f"Invalid projection_type '{projection_type}'. Must be one of {valid_projection_types}"}
    try:
        if zarr_array.ndim != 5:
            logger.warning(f"[Heatmap] Expected 5D array (t,c,z,y,x), but got {zarr_array.ndim}D. Proceeding cautiously.")
        channel_axis = 1
        if not (0 <= channel_index < zarr_array.shape[channel_axis]):
            return {'error': f'Invalid channel_index {channel_index}. Must be between 0 and {zarr_array.shape[channel_axis] - 1}. Array shape: {zarr_array.shape}'}
        max_z_arr, max_y_arr, max_x_arr = zarr_array.shape[2:] # Renamed to avoid conflict
        if not (0 <= roi_z[0] < roi_z[1] <= max_z_arr and
                0 <= roi_y[0] < roi_y[1] <= max_y_arr and # Validation is on original roi_y
                0 <= roi_x[0] < roi_x[1] <= max_x_arr):
            return {'error': f'ROI Z{roi_z}, Y{roi_y}, X{roi_x} is out of bounds for array shape Z{max_z_arr}, Y{max_y_arr}, X{max_x_arr}.'}
        
        total_y_height_in_zarr = zarr_array.shape[3]
        zarr_slice_y_start = total_y_height_in_zarr - roi_y[1]
        zarr_slice_y_end = total_y_height_in_zarr - roi_y[0]
        zarr_slice_y_start = max(0, zarr_slice_y_start)
        zarr_slice_y_end = min(total_y_height_in_zarr, zarr_slice_y_end)

        if zarr_slice_y_start >= zarr_slice_y_end and not (roi_y[0] == 0 and roi_y[1] == total_y_height_in_zarr) :
            return {'error': f'Transformed Y ROI resulted in invalid slice: [{zarr_slice_y_start}, {zarr_slice_y_end}). Original roi_y: {roi_y}, Zarr Y height: {total_y_height_in_zarr}'}
        
        logger.info(f"[Heatmap] Original roi_y (bottom-up): {roi_y}")
        logger.info(f"[Heatmap] Transformed Zarr Y slice (top-down): [{zarr_slice_y_start}, {zarr_slice_y_end})")
        
        roi_slice = (
            slice(0, 1),
            slice(channel_index, channel_index + 1),
            slice(roi_z[0], roi_z[1]),
            slice(zarr_slice_y_start, zarr_slice_y_end),
            slice(roi_x[0], roi_x[1])
        )
        logger.info(f"[Heatmap] Reading slice: Ch={channel_index}, Z_roi={roi_z}, Y_ZARR_SLICE=[{zarr_slice_y_start}:{zarr_slice_y_end}], X_roi={roi_x} from array shape {zarr_array.shape}")
        start_read = time.time()
        data_roi = zarr_array[roi_slice]
        end_read = time.time()
        logger.info(f"[Heatmap] Read in {end_read - start_read:.2f}s. Result shape: {data_roi.shape}, Size: {data_roi.size}")
        if data_roi.size == 0:
            return {'error': 'Specified ROI resulted in an empty data slice.'}
        logger.info(f"[Heatmap] Calculating Z-projection (type='{projection_type}')...")
        start_proj = time.time()
        
        # Convert to float32 for calculations
        data_roi = data_roi.astype(np.float32)
        
        if projection_type == 'mean':
            # Calculate mean along Z axis
            projected_data = np.mean(data_roi, axis=2)
        elif projection_type == 'max':
            projected_data = np.max(data_roi, axis=2)
            
        heatmap_2d = np.squeeze(projected_data, axis=(0, 1))
        
        # Normalize the heatmap to [0, 1]
        min_val = np.min(heatmap_2d)
        max_val = np.max(heatmap_2d)
        if max_val > min_val:
            heatmap_2d = (heatmap_2d - min_val) / (max_val - min_val)
        else:
            heatmap_2d = np.zeros_like(heatmap_2d)  # If all values are the same, return zeros
            
        end_proj = time.time()
        logger.info(f"[Heatmap] Projection calculated in {end_proj - start_proj:.2f}s. Final heatmap shape: {heatmap_2d.shape}")
        return {
            'heatmap': heatmap_2d.tolist(),
            'shape': heatmap_2d.shape,
            'min_val': float(min_val),
            'max_val': float(max_val)
        }
    except IndexError as ie:
         logger.error(f"[Heatmap] IndexError: {ie}", exc_info=True)
         return {'error': f'ROI coordinates likely out of bounds. Details: {ie}'}
    except ValueError as ve:
        logger.error(f"[Heatmap] ValueError: {ve}", exc_info=True)
        return {'error': f'Invalid ROI values? ROI: Z{roi_z}, Y{roi_y}, X{roi_x}. Details: {ve}'}
    except Exception as e:
        logger.error(f"[Heatmap] Unexpected Error: {e}", exc_info=True)
        return {'error': f'Unexpected error during heatmap calculation: {e}'}
# --- End of Analysis Functions ---
# --- Helper function to open the target Zarr array ---
def open_target_zarr_array():
    """Opens and returns the target Zarr array (/0/2). Returns None on failure."""
    target_image_arr = None
    s3_local = None
    try:
        logger.info("Creating S3FileSystem...")
        s3_local = s3fs.S3FileSystem(anon=True)
        logger.info("S3FileSystem created successfully.")
        
        logger.info(f"Creating S3Map for root: {ZARR_BASE_URL}")
        root_store = s3fs.S3Map(root=ZARR_BASE_URL, s3=s3_local, check=False)
        logger.info("S3Map created successfully.")

        logger.info("Opening root Zarr group...")
        try:
            root_group = pyzarr.open_consolidated(store=root_store)
            logger.info("Successfully opened consolidated Zarr group.")
        except Exception as e:
            logger.warning(f"Failed to open consolidated Zarr group: {e}. Trying regular open...")
            root_group = pyzarr.open_group(store=root_store, mode='r')
            logger.info("Successfully opened regular Zarr group.")

        if not root_group:
            logger.error("Failed to open root Zarr group.")
            return None

        logger.info(f"Checking for image group '{ZARR_IMAGE_GROUP_PATH}'...")
        if ZARR_IMAGE_GROUP_PATH not in root_group:
            logger.error(f"Image group path '{ZARR_IMAGE_GROUP_PATH}' not found in root group.")
            logger.info(f"Available groups: {list(root_group.keys())}")
            return None

        image_group_0 = root_group[ZARR_IMAGE_GROUP_PATH]
        logger.info(f"Found image group '{ZARR_IMAGE_GROUP_PATH}'.")

        logger.info(f"Checking for resolution path '{TARGET_RESOLUTION_PATH}'...")
        if TARGET_RESOLUTION_PATH not in image_group_0:
            logger.error(f"Target resolution path '{TARGET_RESOLUTION_PATH}' not found in group '{ZARR_IMAGE_GROUP_PATH}'.")
            logger.info(f"Available resolutions: {list(image_group_0.keys())}")
            return None

        target_image_arr = image_group_0[TARGET_RESOLUTION_PATH]
        logger.info(f"✅ Successfully opened target Zarr array /{ZARR_IMAGE_GROUP_PATH}/{TARGET_RESOLUTION_PATH}")
        logger.info(f"Array shape: {target_image_arr.shape}")
        logger.info(f"Array dtype: {target_image_arr.dtype}")
        return target_image_arr

    except Exception as e:
        logger.error(f" Failed to open target Zarr array: {e}", exc_info=True)
        logger.error(f"State at error: s3_local={'Exists' if s3_local else 'None'}")
        return None
# ---
def generate_vitessce_config(view_state_data):
    """Generates the Vitessce config using base settings and view_state_data from frontend."""
    logger.info(f"Generating Vitessce config with custom view state...")
    if not view_state_data:
        logger.warning("No view_state_data provided, returning minimal config.")
        vc = VitessceConfig(schema_version="1.0.16", name="Error - No View State")
        return vc.to_dict()

    try:
        vc = VitessceConfig(schema_version="1.0.16", name=f"BioMedVis Challenge - Custom View")
        dataset = vc.add_dataset(name="Blood Vessel", uid="bv").add_file(
            url="https://lsp-public-data.s3.amazonaws.com/yapp-2023-3d-melanoma/Dataset1-LSP13626-melanoma-in-situ/0",
            file_type="image.ome-zarr"
        )

        spatial = vc.add_view("spatialBeta", dataset=dataset)
        lc = vc.add_view("layerControllerBeta", dataset=dataset)

        # --- Reconstruct CL objects from view_state_data --- 
        reconstructed_state = {}
        # Copy top-level keys directly
        for key in ["spatialTargetZ", "spatialTargetT", "spatialZoom", "spatialTargetX", "spatialTargetY", "spatialRenderingMode"]:
            if key in view_state_data:
                reconstructed_state[key] = view_state_data[key]
        
        # Process imageLayer - needs CL wrapper for the list
        if "imageLayer" in view_state_data and isinstance(view_state_data["imageLayer"], list):
            processed_layers = []
            for layer_data in view_state_data["imageLayer"]:
                processed_layer = layer_data.copy()
                # Process imageChannel within the layer - needs CL wrapper for the list
                if "imageChannel" in processed_layer and isinstance(processed_layer["imageChannel"], list):
                    # Convert Python boolean to JavaScript boolean
                    for channel in processed_layer["imageChannel"]:
                        if "spatialChannelVisible" in channel:
                            channel["spatialChannelVisible"] = bool(channel["spatialChannelVisible"])
                    processed_layer["imageChannel"] = CL(processed_layer["imageChannel"])
                else:
                    processed_layer["imageChannel"] = CL([])
                processed_layers.append(processed_layer)
            reconstructed_state["imageLayer"] = CL(processed_layers)
        else:
            reconstructed_state["imageLayer"] = CL([])

        # Link views using the reconstructed state
        scope = get_initial_coordination_scope_prefix("bv", "image")
        logger.info(f"Linking views with reconstructed state: {reconstructed_state}")
        vc.link_views_by_dict([spatial, lc], reconstructed_state, meta=True, scope_prefix=scope)
        
        # Define layout
        vc.layout(spatial | lc)

        config_dict = vc.to_dict()
        logger.info(f" Vitessce Configuration generated successfully from POST data.")
        return config_dict
    except Exception as e:
        logger.error(f"Error generating Vitessce config from POST data: {e}", exc_info=True)
        return None
# --- New POST Endpoint --- 
@app.route('/api/generate_config', methods=['POST'])
def generate_config_from_post():
    """Generates Vitessce config based on view state data sent in POST request body."""
    logger.info("Request received for /api/generate_config [POST]")
    view_state_data = request.get_json()
    if not view_state_data:
        logger.error("No JSON data received in POST request.")
        return jsonify({"error": "No JSON data received in POST request body"}), 400
    
    logger.debug(f"Received view_state_data: {view_state_data}")

    config = generate_vitessce_config(view_state_data=view_state_data)
    if config:
        return jsonify(config)
    else:
        return jsonify({"error": "Failed to generate Vitessce configuration from provided data"}), 500
# --- New Zarr Proxy Endpoint --- 
@app.route('/api/zarr-proxy')
def zarr_proxy():
    """Proxies requests to a Zarr URL to bypass CORS issues."""
    target_url = request.args.get('url')
    if not target_url:
        return jsonify({"error": "Missing 'url' query parameter"}), 400

    logger.info(f"Proxying request for Zarr URL: {target_url}")

    try:
        # Make the request to the target URL, stream the response
        # Use HEAD request first for metadata, then GET if needed? Zarr library might handle this.
        # For simplicity, directly proxy GET for now.
        # Pass through necessary headers if zarr library requires them (e.g., Range)
        # Note: This simple proxy doesn't handle Range requests yet, which might be needed for Zarr chunks.
        # A more robust proxy would parse Range headers and forward them.

        # Send the request to the target URL
        resp = requests.get(target_url, stream=True, headers=request.headers)
        resp.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx)

        # Stream the response back to the client
        # Preserve content type from the original response
        response_headers = {
            'Content-Type': resp.headers.get('Content-Type', 'application/octet-stream'),
            # Add other headers if needed, carefully consider security implications
        }

        # Use stream_with_context to efficiently stream large Zarr chunks
        return Response(stream_with_context(resp.iter_content(chunk_size=8192)),
                        status=resp.status_code,
                        headers=response_headers)

    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching Zarr URL {target_url}: {e}")
        return jsonify({"error": f"Failed to fetch Zarr URL: {e}"}), 502 # Bad Gateway
    except Exception as e:
        logger.error(f"Unexpected error in Zarr proxy: {e}", exc_info=True)
        return jsonify({"error": "Internal server error in proxy"}), 500
# --- End of New Zarr Proxy Endpoint --- 
@app.route('/api/roi_shapes', methods=['GET'])
def serve_roi_shapes():
    try:
        roi_path = Path(__file__).parent / "output" / "roi_shapes.json"  # <-- FIXED
        if not roi_path.exists():
            logger.warning("ROI shapes file not found.")
            return jsonify({"error": "ROI shapes not found"}), 404

        with open(roi_path, "r") as f:
            roi_json = json.load(f)

        logger.info(" ROI shapes successfully served.")
        return jsonify(roi_json)

    except Exception as e:
        logger.error(f" Error serving ROI shapes: {e}", exc_info=True)
        return jsonify({"error": "Failed to load ROI shapes"}), 500
if __name__ == '__main__':
    logger.info("Starting Flask backend server on http://127.0.0.1:5000")
    # Port 5000 is common for backend APIs
    # host='0.0.0.0' makes it accessible from other containers/machines if needed
    app.run(host='127.0.0.1', port=5000, debug=True) 