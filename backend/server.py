# Backend server code will go here 
import json
import logging
from flask import Flask, jsonify, request, Response, stream_with_context, send_file
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
# Enable CORS for all routes during development
CORS(app, resources={r"/*": {"origins": "*"}})

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
            logger.info(f" Successfully extracted channel names: {channel_map}")
            return jsonify(channel_map)
        else:
            logger.warning("Could not find valid 'omero' metadata with 'channels'. Returning 404.")
            # Return empty map or error?
            return jsonify({"error": "Channel metadata ('omero' with 'channels') not found in Zarr store"}), 404

    except Exception as e:
        logger.error(f" Failed during channel name retrieval: {e}", exc_info=True)
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

        # Check if circle file exists and add it to the dataset
        output_dir = Path(__file__).parent / "output"
        circle_file = output_dir / "roi_circles.geojson"
        if circle_file.exists():
            logger.info("Adding circle layer to dataset")
            dataset.add_file(
                url="http://localhost:5000/api/roi_circles",
                file_type="obsSegmentations.json",
                options={
                    "mapping": [
                        {
                            "key": "fillColor",
                            "value": "properties.fillColor"
                        },
                        {
                            "key": "fillOpacity",
                            "value": "properties.fillOpacity"
                        },
                        {
                            "key": "strokeColor",
                            "value": "properties.strokeColor"
                        },
                        {
                            "key": "strokeWidth",
                            "value": "properties.strokeWidth"
                        },
                        {
                            "key": "strokeOpacity",
                            "value": "properties.strokeOpacity"
                        }
                    ]
                }
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
        roi_path = Path(__file__).parent / "output" / "roi_shapes.geojson"  # Changed from .json to .geojson
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

#########################################Histograms#########################################

def calculate_histogram(zarr_array, channel_index, roi_z, roi_y, roi_x, num_bins=50):
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


def calculate_z_projection_heatmap(zarr_array, channel_index, roi_z, roi_y, roi_x, projection_type='mean'):
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
        
        # Log array shape and ROI values
        logger.info(f"Array shape: {zarr_array.shape}")
        logger.info(f"ROI values - Z: {roi_z}, Y: {roi_y}, X: {roi_x}")
        
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
        
        logger.info(f"[Heatmap] Original roi_y (bottom-up): {roi_y}")
        logger.info(f"[Heatmap] Transformed Zarr Y slice (top-down): [{zarr_slice_y_start}, {zarr_slice_y_end})")
        
        roi_slice = (
            slice(0, 1),
            slice(channel_index, channel_index + 1),
            slice(roi_z[0], roi_z[1]),
            slice(zarr_slice_y_start, zarr_slice_y_end),
            slice(roi_x[0], roi_x[1])
        )
        
        logger.info(f"[Heatmap] Reading slice: Ch={channel_index}, Z_roi={roi_z}, Y_ZARR_SLICE=[{zarr_slice_y_start}:{zarr_slice_y_end}], X_roi={roi_x}")
        start_read = time.time()
        
        try:
            data_roi = zarr_array[roi_slice]
            end_read = time.time()
            logger.info(f"[Heatmap] Read in {end_read - start_read:.2f}s. Result shape: {data_roi.shape}, Size: {data_roi.size}")
            
            if data_roi.size == 0:
                return {'error': 'Specified ROI resulted in an empty data slice.'}
            
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
                heatmap_2d = np.zeros_like(heatmap_2d)
            
            logger.info(f"[Heatmap] Final heatmap shape: {heatmap_2d.shape}, Range: [{np.min(heatmap_2d):.3f}, {np.max(heatmap_2d):.3f}]")
            
            return {
                'heatmap': heatmap_2d.tolist(),
                'shape': heatmap_2d.shape,
                'min_val': float(min_val),
                'max_val': float(max_val)
            }
            
        except Exception as read_error:
            logger.error(f"[Heatmap] Error reading data slice: {str(read_error)}")
            return {'error': f'Error reading data slice: {str(read_error)}'}
            
    except Exception as e:
        logger.error(f"[Heatmap] Unexpected Error: {e}", exc_info=True)
        return {'error': f'Unexpected error during heatmap calculation: {e}'}

# Helper function for normalization
def normalize_array(arr):
    """Normalizes a numpy array to the range [0, 1]."""
    min_val = np.min(arr)
    max_val = np.max(arr)
    if max_val == min_val:
        # Handle constant array case (avoid division by zero)
        # Return array of zeros or maybe scaled based on the constant value?
        # Returning zeros is common for visualization range mapping.
        return np.zeros_like(arr, dtype=np.float32)
    else:
        return (arr - min_val) / (max_val - min_val)

# --- Modified API Endpoint for Z-Projection Heatmap (Single or Interaction) ---
@app.route('/api/z_projection', methods=['POST'])
def get_z_projection_data():
    """
    Calculates and returns Z-projection heatmap data.
    Can return a single channel projection or a combined interaction heatmap
    based on the sum of normalized projections of multiple channels.
    """
    logger.info("Request received for /api/z_projection")
    data = request.get_json()
    if not data:
        logger.error("No JSON data received for z_projection.")
        return jsonify({"error": "Missing JSON payload"}), 400

    # Extract parameters
    channel_indices = data.get('channel_indices') # Expecting a list of integers
    roi = data.get('roi') # Expecting {'z': [min, max], 'y': [min, max], 'x': [min, max]} or None
    is_interaction = data.get('interaction', False) # Boolean, defaults to False
    projection_type = data.get('projection_type', 'mean') # 'mean' or 'max'

    # Validate parameters
    if not channel_indices or not isinstance(channel_indices, list) or not all(isinstance(i, int) for i in channel_indices):
        logger.error(f"Invalid or missing 'channel_indices': {channel_indices}. Must be a non-empty list of integers.")
        return jsonify({"error": "Invalid or missing 'channel_indices'. Must be a non-empty list of integers."}), 400

    if is_interaction:
        if not (2 <= len(channel_indices) <= 6):
            logger.error(f"Interaction heatmap requires 2 to 6 channels, but got {len(channel_indices)}.")
            return jsonify({"error": f"Interaction heatmap requires 2 to 6 channels, got {len(channel_indices)}."}), 400
        logger.info(f"Interaction Heatmap requested for Channels: {channel_indices}, ROI: {roi}, Proj: {projection_type}")
    else:
        # For single heatmap, we only use the first index, but log all received for clarity
        logger.info(f"Single Heatmap requested for Channel: {channel_indices[0]} (from list {channel_indices}), ROI: {roi}, Proj: {projection_type}")


    # Validate projection_type
    if projection_type not in ['mean', 'max']:
         logger.error(f"Invalid projection_type: {projection_type}. Must be 'mean' or 'max'.")
         return jsonify({"error": f"Invalid projection_type: {projection_type}. Must be 'mean' or 'max'."}), 400

    # Validate ROI if provided
    valid_roi = None
    if roi:
        if not all(k in roi for k in ('z', 'y', 'x')) or \
           not all(isinstance(roi[k], list) and len(roi[k]) == 2 for k in ('z', 'y', 'x')):
            logger.error(f"Invalid 'roi' format: {roi}.")
            return jsonify({"error": "Invalid 'roi' format. Expected {'z': [start, end], 'y': [start, end], 'x': [start, end]} or null"}), 400
        try:
            # Ensure ROI values are integers
            valid_roi = {
                'z': (int(roi['z'][0]), int(roi['z'][1])),
                'y': (int(roi['y'][0]), int(roi['y'][1])),
                'x': (int(roi['x'][0]), int(roi['x'][1])),
            }
            logger.info(f"Using ROI: {valid_roi}")
        except (ValueError, TypeError) as e:
             logger.error(f"Invalid numeric value or type in ROI: {roi}, Error: {e}")
             return jsonify({"error": "Invalid numeric value or type in ROI components."}), 400
    else:
        logger.info("No ROI provided, processing full channel extent (will be determined after opening array).")


    # --- Open the target Zarr array ---
    target_zarr_array = open_target_zarr_array()
    if target_zarr_array is None:
        return jsonify({"error": "Failed to open target Zarr array on server."}), 500
    # --

    # Determine full extent if ROI was not provided
    shape = target_zarr_array.shape
    if not valid_roi:
        # Assuming 5D T,C,Z,Y,X - adjust if different
        if len(shape) == 5:
            z_max, y_max, x_max = shape[2:]
        elif len(shape) == 4: # Assuming C,Z,Y,X
            z_max, y_max, x_max = shape[1:]
        else:
             logger.error(f"Unsupported array dimension: {len(shape)} found in Zarr array.")
             return jsonify({"error": f"Unsupported Zarr array dimension {len(shape)}."}), 500
        valid_roi = {'z': (0, z_max), 'y': (0, y_max), 'x': (0, x_max)}
        logger.info(f"Using full array extent as ROI: {valid_roi}")

    # Extract final ROI tuples
    roi_z = tuple(valid_roi['z'])
    roi_y = tuple(valid_roi['y'])
    roi_x = tuple(valid_roi['x'])


    try:
        if not is_interaction:
            # --- Single Channel Heatmap ---
            target_channel = channel_indices[0]
            logger.info(f"Calculating single {projection_type} projection for channel {target_channel}...")

            heatmap_result = calculate_z_projection_heatmap(
                zarr_array=target_zarr_array,
                channel_index=target_channel,
                roi_z=roi_z,
                roi_y=roi_y,
                roi_x=roi_x,
                projection_type=projection_type
            )

            if 'error' in heatmap_result:
                logger.error(f"Error from calculate_z_projection_heatmap: {heatmap_result['error']}")
                return jsonify({"error": f"Heatmap calculation failed: {heatmap_result['error']}"}), 500
            else:
                logger.info(f"✅ Single heatmap calculated successfully for channel {target_channel}.")
                # Return only the heatmap data and shape
                return jsonify({
                    "heatmap": heatmap_result.get('heatmap', []),
                    "shape": heatmap_result.get('shape')
                    }), 200

        else:
            # --- Interaction Heatmap ---
            logger.info(f"Calculating interaction heatmap for channels {channel_indices}...")
            total_heatmap_norm = None
            final_shape = None

            for i, channel_index in enumerate(channel_indices):
                logger.info(f"  Processing channel {channel_index} ({i+1}/{len(channel_indices)})...")
                heatmap_result = calculate_z_projection_heatmap(
                    zarr_array=target_zarr_array,
                    channel_index=channel_index,
                    roi_z=roi_z,
                    roi_y=roi_y,
                    roi_x=roi_x,
                    projection_type=projection_type # Use the same projection for all
                )

                if 'error' in heatmap_result:
                    logger.error(f"Error calculating heatmap for channel {channel_index}: {heatmap_result['error']}")
                    # Fail the whole interaction calculation if one channel fails
                    return jsonify({"error": f"Interaction heatmap failed on channel {channel_index}: {heatmap_result['error']}"}), 500

                heatmap_raw = np.array(heatmap_result.get('heatmap'), dtype=np.float32)
                current_shape = heatmap_result.get('shape')

                if heatmap_raw.size == 0:
                    logger.warning(f"  Skipping channel {channel_index} - resulted in empty heatmap.")
                    continue # Skip to next channel

                # Store shape from the first valid heatmap
                if final_shape is None and current_shape:
                    final_shape = current_shape

                # Normalize the current heatmap
                heatmap_norm = normalize_array(heatmap_raw)

                # Add to the total accumulator
                if total_heatmap_norm is None:
                    total_heatmap_norm = heatmap_norm
                else:
                    # Ensure shapes match before adding (should always match if ROI is the same)
                    if total_heatmap_norm.shape != heatmap_norm.shape:
                        logger.error(f"Shape mismatch! Acc: {total_heatmap_norm.shape}, Current: {heatmap_norm.shape} for Ch: {channel_index}")
                        return jsonify({"error": f"Internal error: Shape mismatch during interaction heatmap accumulation."}), 500
                    total_heatmap_norm += heatmap_norm

                logger.info(f"  Channel {channel_index} processed and added.")

            if total_heatmap_norm is None:
                logger.error("Interaction heatmap calculation resulted in no data (all channels failed or were empty).")
                return jsonify({"error": "Interaction heatmap failed: No data processed."}), 500

            logger.info(f"✅ Interaction heatmap calculated successfully for channels {channel_indices}.")
            # Return the combined heatmap and its shape
            return jsonify({
                "heatmap": total_heatmap_norm.tolist(),
                "shape": final_shape
                }), 200

    except MemoryError:
        logger.error(f" MemoryError while processing Z-projection. Data might be too large.", exc_info=True)
        return jsonify({"error": "Server ran out of memory processing the data."}), 500
    except Exception as e:
        logger.error(f" Unexpected error in /api/z_projection route: {e}", exc_info=True)
        return jsonify({"error": f"Internal server error: {e}"}), 500


def calculate_rgb_interaction_heatmap_py(zarr_array, roi_z, roi_y, roi_x, projection_type='mean'):
    if zarr_array is None:
        return {'error': 'Zarr array is None.'}
    valid_projection_types = ['mean', 'max']
    if projection_type not in valid_projection_types:
        return {'error': f"Invalid projection_type '{projection_type}'. Must be one of {valid_projection_types}"}
    try:
        if zarr_array.ndim != 5:
            logger.warning(f"[Interaction Heatmap] Expected 5D array (t,c,z,y,x), but got {zarr_array.ndim}D. Proceeding cautiously.")
        
        # Define channel groups and their ranges
        channel_groups = {
            1: [19, 37],      # group 1: CD31 + CD11b
            2: [37, 59],      # group 2: CD11b + Catalase
            3: [19, 25, 27],  # group 3: CD31 + CD4 + CD20
            4: [25, 27]       # group 4: CD4 + CD20
        }
        
        # Define channel ranges
        channel_ranges = {
            19: [300, 20000],    # CD31
            27: [1000, 7000],    # CD20
            37: [700, 6000],     # CD11b
            25: [1638, 10000],   # CD4
            59: [1638, 7000]     # Catalase
        }
        
        logger.info(f"Processing channel groups: {channel_groups}")
        
        channel_axis = 1
        for group_val, channels in channel_groups.items():
            for ch_idx in channels:
                if not (0 <= ch_idx < zarr_array.shape[channel_axis]):
                    return {'error': f'Invalid channel_index {ch_idx} in group {group_val}. Must be between 0 and {zarr_array.shape[channel_axis] - 1}.'}
        
        max_z_arr, max_y_arr, max_x_arr = zarr_array.shape[2:]
        if not (0 <= roi_z[0] < roi_z[1] <= max_z_arr and 0 <= roi_y[0] < roi_y[1] <= max_y_arr and 0 <= roi_x[0] < roi_x[1] <= max_x_arr):
            return {'error': f'ROI Z{roi_z}, Y{roi_y}, X{roi_x} is out of bounds for array shape Z{max_z_arr}, Y{max_y_arr}, X{max_x_arr}.'}
        
        total_y_height_in_zarr = zarr_array.shape[3]
        zarr_slice_y_start = total_y_height_in_zarr - roi_y[1]
        zarr_slice_y_end = total_y_height_in_zarr - roi_y[0]
        zarr_slice_y_start = max(0, zarr_slice_y_start)
        zarr_slice_y_end = min(total_y_height_in_zarr, zarr_slice_y_end)
        
        if zarr_slice_y_start >= zarr_slice_y_end and not (roi_y[0] == 0 and roi_y[1] == total_y_height_in_zarr):
            return {'error': f'Transformed Y ROI resulted in invalid slice: [{zarr_slice_y_start}, {zarr_slice_y_end}). Original roi_y: {roi_y}, Zarr Y height: {total_y_height_in_zarr}'}
        
        group_maps = {}
        for group_val, channels in channel_groups.items():
            logger.info(f"\nProcessing group {group_val} with channels {channels}")
            channel_heatmaps = []
            channel_stats = []
            
            for ch_idx in channels:
                logger.info(f"\n  Processing channel {ch_idx}")
                roi_slice = (
                    slice(0, 1),
                    slice(ch_idx, ch_idx + 1),
                    slice(roi_z[0], roi_z[1]),
                    slice(zarr_slice_y_start, zarr_slice_y_end),
                    slice(roi_x[0], roi_x[1])
                )
                data_roi = zarr_array[roi_slice]
                
                # Apply channel-specific range filtering first
                if ch_idx in channel_ranges:
                    min_range, max_range = channel_ranges[ch_idx]
                    data_roi = np.clip(data_roi, min_range, max_range)
                
                # Find maximum value in the entire channel data
                max_val = np.max(data_roi)
                logger.info(f"  Channel {ch_idx} max value: {max_val:.3f}")
                
                # Normalize all values by the maximum
                data_roi = data_roi / max_val
                
                # Calculate mean along Z axis for each x,y position
                heatmap_2d = np.mean(data_roi, axis=2)
                heatmap_2d = np.squeeze(heatmap_2d, axis=(0, 1))
                
                # Apply threshold - set values below 0.1 to 0
                min_threshold = 0.1
                heatmap_2d[heatmap_2d < min_threshold] = 0
                
                # Log stats after processing
                logger.info(f"  After processing:")
                logger.info(f"    Range: [{np.min(heatmap_2d):.3f}, {np.max(heatmap_2d):.3f}]")
                logger.info(f"    Mean: {np.mean(heatmap_2d):.3f}")
                logger.info(f"    Values near 1 (>0.9): {np.sum(heatmap_2d > 0.9)}")
                logger.info(f"    Non-zero values: {np.sum(heatmap_2d > 0)}")
                
                channel_heatmaps.append(heatmap_2d)
                channel_stats.append({
                    'max': np.max(heatmap_2d),
                    'mean': np.mean(heatmap_2d),
                    'non_zero': np.sum(heatmap_2d > 0)
                })
            
            # Calculate interaction as mean of channel heatmaps
            # Stack all channel heatmaps
            stacked_heatmaps = np.stack(channel_heatmaps)
            
            # Calculate mean along channel axis (axis=0)
            interaction = np.mean(stacked_heatmaps, axis=0)
            
            # Normalize interaction values to [0, 1]
            min_val = np.min(interaction)
            max_val = np.max(interaction)
            if max_val > min_val:
                interaction = (interaction - min_val) / (max_val - min_val)
            else:
                interaction = np.zeros_like(interaction)
            
            # Log statistics about the interaction values
            non_zero_mask = interaction > 0
            if np.any(non_zero_mask):
                non_zero_values = interaction[non_zero_mask]
                logger.info(f"\nGroup {group_val} interaction statistics:")
                logger.info(f"  Channel stats:")
                for i, stats in enumerate(channel_stats):
                    logger.info(f"    Channel {channels[i]}:")
                    logger.info(f"      Max: {stats['max']:.3f}")
                    logger.info(f"      Mean: {stats['mean']:.3f}")
                    logger.info(f"      Non-zero: {stats['non_zero']}")
                logger.info(f"  Interaction stats:")
                logger.info(f"    Total pixels: {interaction.size}")
                logger.info(f"    Non-zero pixels: {np.sum(non_zero_mask)}")
                logger.info(f"    Values near 1 (>0.9): {np.sum(non_zero_values > 0.9)}")
                logger.info(f"    Values near 0 (<0.1): {np.sum(non_zero_values < 0.1)}")
                logger.info(f"    Mean value: {np.mean(non_zero_values):.3f}")
                logger.info(f"    Median value: {np.median(non_zero_values):.3f}")
                logger.info(f"    Max value: {np.max(non_zero_values):.3f}")
                logger.info(f"    Min value: {np.min(non_zero_values):.3f}")
            
            group_maps[f'group_{group_val}'] = interaction.tolist()
        
        height, width = next(iter(group_maps.values())).__len__(), next(iter(group_maps.values()))[0].__len__()
        logger.info(f"\nAll groups processed. Final heatmap dimensions: {height}x{width}")
        
        return {
            'heatmaps': group_maps,
            'shape': [height, width]
        }
    except Exception as e:
        logger.error(f"[Interaction Heatmap] Error: {e}", exc_info=True)
        return {'error': f'Error calculating interaction heatmap: {e}'}
# --- API Endpoint for RGB Interaction Heatmap ---
@app.route('/api/rgb_interaction_heatmap', methods=['POST'])
def get_rgb_interaction_heatmap_data():
    """API endpoint to get interaction heatmap data."""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        roi = data.get('roi')
        if not roi or not all(k in roi for k in ['z', 'y', 'x']):
            return jsonify({'error': 'Invalid ROI format. Must include z, y, x ranges.'}), 400
        
        projection_type = data.get('projection_type', 'mean')
        
        # Get the Zarr array
        zarr_array = open_target_zarr_array()
        if zarr_array is None:
            return jsonify({'error': 'Failed to open Zarr array'}), 500
        
        # Calculate the interaction heatmap
        result = calculate_rgb_interaction_heatmap_py(
            zarr_array,
            roi['z'],
            roi['y'],
            roi['x'],
            projection_type
        )
        
        if 'error' in result:
            return jsonify(result), 400
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error in get_rgb_interaction_heatmap_data: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@app.route('/api/analyze_heatmaps', methods=['POST'])
def analyze_heatmaps():
    """API endpoint to analyze heatmaps for specific channels in a given ROI."""
    try:
        data = request.get_json()
        if not data:
            logger.error("No data provided in request")
            return jsonify({'error': 'No data provided'}), 400
        
        roi = data.get('roi')
        if not roi or not all(k in roi for k in ['xMin', 'xMax', 'yMin', 'yMax', 'zMin', 'zMax']):
            logger.error(f"Invalid ROI format: {roi}")
            return jsonify({'error': 'Invalid ROI format'}), 400

        channels = data.get('channels', ['CD31', 'CD11b', 'Catalase', 'CD4', 'CD20', 'CD11c'])
        if not channels:
            logger.error("No channels specified")
            return jsonify({'error': 'No channels specified'}), 400
        
        # Get the Zarr array
        zarr_array = open_target_zarr_array()
        if zarr_array is None:
            logger.error("Failed to open Zarr array")
            return jsonify({'error': 'Failed to open Zarr array'}), 500
        
        logger.info(f"Processing ROI: {roi}")
        logger.info(f"Processing channels: {channels}")
        logger.info(f"Zarr array shape: {zarr_array.shape}")
        
        # Validate ROI values
        try:
            roi_values = {
                'zMin': int(roi['zMin']),
                'zMax': int(roi['zMax']),
                'yMin': int(roi['yMin']),
                'yMax': int(roi['yMax']),
                'xMin': int(roi['xMin']),
                'xMax': int(roi['xMax'])
            }
        except ValueError as ve:
            logger.error(f"Invalid ROI values: {ve}")
            return jsonify({'error': 'Invalid ROI values - all values must be integers'}), 400
        
        # Define channel mapping
        channel_mapping = {
            'CD31': 19,
            'CD11b': 37,
            'Catalase': 59,
            'CD4': 25,
            'CD20': 27,
            'CD11c': 41
        }
        
        # Process only specified channels
        results = {}
        for channel_name in channels:
            if channel_name not in channel_mapping:
                logger.warning(f"Skipping unknown channel: {channel_name}")
                continue
                
            channel_index = channel_mapping[channel_name]
            logger.info(f"Processing channel {channel_name} (index: {channel_index})")
            
            try:
                heatmap_result = calculate_z_projection_heatmap(
                    zarr_array=zarr_array,
                    channel_index=channel_index,
                    roi_z=(roi_values['zMin'], roi_values['zMax']),
                    roi_y=(roi_values['yMin'], roi_values['yMax']),
                    roi_x=(roi_values['xMin'], roi_values['xMax']),
                    projection_type='mean'
                )
                
                if 'error' not in heatmap_result:
                    results[channel_name] = heatmap_result.get('heatmap', [])
                    logger.info(f"Successfully processed channel {channel_name}")
                else:
                    logger.error(f"Error processing channel {channel_name}: {heatmap_result['error']}")
            except Exception as channel_error:
                logger.error(f"Error processing channel {channel_name}: {str(channel_error)}", exc_info=True)
                continue
        
        if not results:
            logger.error("No results were generated for any channel")
            return jsonify({'error': 'Failed to generate heatmaps for any channel'}), 500
            
        logger.info(f"Returning results for {len(results)} channels")
        return jsonify(results)
        
    except Exception as e:
        logger.error(f"Error in analyze_heatmaps: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@app.route('/api/analyze_interaction_heatmap', methods=['POST'])
def analyze_interaction_heatmap():
    """API endpoint to analyze interaction heatmaps for defined groups in a given ROI."""
    try:
        data = request.get_json()
        if not data:
            logger.error("No data provided in request")
            return jsonify({'error': 'No data provided'}), 400
        
        roi = data.get('roi')
        if not roi or not all(k in roi for k in ['xMin', 'xMax', 'yMin', 'yMax', 'zMin', 'zMax']):
            logger.error(f"Invalid ROI format: {roi}")
            return jsonify({'error': 'Invalid ROI format'}), 400
        
        # Get the Zarr array
        zarr_array = open_target_zarr_array()
        if zarr_array is None:
            logger.error("Failed to open Zarr array")
            return jsonify({'error': 'Failed to open Zarr array'}), 500
        
        logger.info(f"Processing ROI for interaction analysis: {roi}")
        
        # Convert ROI format to match the expected format
        roi_formatted = {
            'z': [int(roi['zMin']), int(roi['zMax'])],
            'y': [int(roi['yMin']), int(roi['yMax'])],
            'x': [int(roi['xMin']), int(roi['xMax'])]
        }
        
        # Calculate interaction heatmap
        result = calculate_rgb_interaction_heatmap_py(
            zarr_array,
            roi_formatted['z'],
            roi_formatted['y'],
            roi_formatted['x'],
            'mean'
        )
        
        if 'error' in result:
            logger.error(f"Error in interaction heatmap calculation: {result['error']}")
            return jsonify(result), 400
            
        logger.info("Successfully calculated interaction heatmaps")
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error in analyze_interaction_heatmap: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@app.route('/api/evaluate_rois', methods=['POST'])
def evaluate_rois():
    """
    API endpoint for evaluating ROI detection performance using a multi-step self-supervised approach.
    
    The evaluation process includes:
    1. Identifying surrogate positive/negative regions based on interaction scores
    2. Training and evaluating a classifier on ROI features
    3. Visualizing ROI embeddings using t-SNE
    4. Checking biological plausibility through biomarker co-expression
    5. Evaluating visual interpretability of ROI overlays
    
    Expected JSON payload:
    {
        'interaction_scores': Array of interaction scores for each region,
        'biomarker_data': Dictionary of biomarker expression data,
        'roi_features': Feature matrix for each ROI,
        'roi_overlays': Optional ROI overlay data,
        'microscopy_images': Optional original microscopy images
    }
    
    Returns:
        JSON object containing evaluation metrics and visualization data
    """
    try:
        data = request.json
        interaction_scores = np.array(data['interaction_scores'])
        biomarker_data = data['biomarker_data']
        roi_features = np.array(data['roi_features'])
        
        # Initialize evaluator with interaction scores and biomarker data
        evaluator = ROIEvaluator(interaction_scores, biomarker_data)
        
        # Step 1: Identify surrogate positive and negative regions
        surrogate_positives, surrogate_negatives = evaluator.identify_surrogate_labels()
        
        # Step 2: Train classifier and evaluate its performance
        clf = evaluator.train_classifier(roi_features)
        classifier_metrics = evaluator.evaluate_classifier(clf, roi_features)
        
        # Step 3: Generate t-SNE visualization of ROI embeddings
        embeddings_2d = evaluator.visualize_embeddings(roi_features)
        
        # Step 4: Analyze biological plausibility through biomarker co-expression
        biological_metrics = evaluator.check_biological_plausibility()
        
        # Step 5: Evaluate visual interpretability if overlay data is provided
        visual_metrics = evaluator.evaluate_visual_interpretability(
            data.get('roi_overlays'),
            data.get('microscopy_images')
        )
        
        # Compile all evaluation results
        results = {
            'classifier_metrics': classifier_metrics,
            'biological_metrics': biological_metrics,
            'visual_metrics': visual_metrics,
            'embeddings_2d': embeddings_2d.tolist()
        }
        
        return jsonify(results)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/config')
def get_config():
    config_path = Path(__file__).parent / "output" / "vitnesse_config.json"
    if not config_path.exists():
        return jsonify({"error": "Configuration file not found"}), 404
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    return jsonify(config)

@app.route('/api/update_config_with_circles', methods=['POST'])
def update_config_with_circles():
    """Update Vitessce config to include circle overlays for ROIs"""
    try:
        data = request.get_json()
        show_circles = data.get('showCircles', False)
        interaction = data.get('interaction', None)
        
        logger.info(f"Updating config with circles: show_circles={show_circles}, interaction={interaction}")
        
        if show_circles and interaction:
            # Load ROI data for the specific interaction
            output_dir = Path(__file__).parent / "output"
            roi_file = output_dir / f"top_roi_scores_{interaction}.json"
            
            if not roi_file.exists():
                return jsonify({"error": f"ROI file not found for interaction: {interaction}"}), 404
            
            with open(roi_file, 'r') as f:
                roi_data = json.load(f)
            
            # Get top 5 ROIs
            top_5_rois = roi_data['top_rois'][:5]
            
            # Create circle features for GeoJSON
            circle_features = []
            y_max = 5508  # Full height of the image
            
            for i, roi in enumerate(top_5_rois):
                # Scale coordinates to fit on screen
                # Assuming screen width is around 1920px and height around 1080px
                screen_width = 1920
                screen_height = 1080
                
                # Scale coordinates to fit within screen bounds
                x = (float(roi['position']['x']) * 8) % screen_width
                y = centroid[1]  # Keep original Y coordinate from GeoJSON
                
                # Ensure circles are visible within viewport
                x = max(50, min(screen_width - 50, x))
                y = max(50, min(screen_height - 50, y))
                
                logger.info(f"ROI {i}: Original centroid: {roi['position']}, Scaled: x={x}, y={y}")
                
                feature = {
                    "type": "Feature",
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [roi['position']]
                    },
                    "properties": {
                        "name": f"ROI_{interaction}_{i}",
                        "interaction": interaction,
                        "roi_id": roi['roi_id'],
                        "score": roi['scores']['combined_score'],
                        "fillColor": get_circle_color(i),
                        "fillOpacity": 0.2,
                        "strokeColor": get_circle_color(i),
                        "strokeWidth": 3,
                        "strokeOpacity": 0.8,
                        "clickable": True,
                        "tooltip": f"ROI {i+1}: Score {roi['scores']['combined_score']:.3f}",
                        "radius": 500,
                        "center": [x, y]
                    }
                }
                circle_features.append(feature)
            
            # Save circle GeoJSON
            circle_geojson = {
                "type": "FeatureCollection",
                "features": circle_features,
                "properties": {
                    "interaction_type": interaction,
                    "total_rois": len(circle_features),
                    "description": f"ROI circles for {interaction} interaction"
                }
            }
            
            circle_output_path = output_dir / "roi_circles.geojson"
            with open(circle_output_path, 'w') as f:
                json.dump(circle_geojson, f, indent=2)
            
            logger.info(f"Created circle GeoJSON with {len(circle_features)} circles")
            return jsonify({
                "success": True,
                "circles_created": len(circle_features),
                "file_path": str(circle_output_path),
                "interaction": interaction
            })
            
        else:
            # Remove circle file if it exists
            output_dir = Path(__file__).parent / "output"
            circle_output_path = output_dir / "roi_circles.geojson"
            if circle_output_path.exists():
                circle_output_path.unlink()
            
            return jsonify({
                "success": True,
                "circles_removed": True
            })
            
    except Exception as e:
        logger.error(f"Error updating config with circles: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

@app.route('/api/roi_circles', methods=['GET'])
def serve_roi_circles():
    """Serve the ROI circles GeoJSON file"""
    try:
        output_dir = Path(__file__).parent / "output"
        circle_file = output_dir / "roi_circles.geojson"
        
        if circle_file.exists():
            with open(circle_file, 'r') as f:
                circle_data = json.load(f)
            return jsonify(circle_data)
        else:
            return jsonify({"type": "FeatureCollection", "features": []})
            
    except Exception as e:
        logger.error(f"Error serving ROI circles: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

@app.route('/api/filtered_rois', methods=['POST'])
def get_filtered_rois():
    """Get ROIs filtered by interaction type for interactive circles"""
    try:
        data = request.get_json()
        interaction_types = data.get('interactions', [])
        
        logger.info(f"Filtering ROIs for interactions: {interaction_types}")
        
        # Load ROI data
        output_dir = Path(__file__).parent / "output"
        roi_file = output_dir / "roi_shapes.geojson"
        
        if not roi_file.exists():
            return jsonify({"error": "ROI shapes file not found"}), 404
        
        with open(roi_file, 'r') as f:
            roi_data = json.load(f)
        
        if not roi_data.get('features'):
            return jsonify({"error": "No ROI features found"}), 404
        
        # Filter ROIs by interaction type
        filtered_features = []
        for feature in roi_data['features']:
            if not feature.get('properties', {}).get('interactions'):
                continue
                
            roi_interactions = feature['properties']['interactions']
            if isinstance(roi_interactions, str):
                roi_interactions = [roi_interactions]
            
            # Check if any of the ROI interactions match the requested types
            if any(interaction in interaction_types for interaction in roi_interactions):
                filtered_features.append(feature)
        
        # Sort by score and take top 5
        filtered_features.sort(
            key=lambda x: x.get('properties', {}).get('score', 0),
            reverse=True
        )
        
        top_5_features = filtered_features[:5]
        
        # Convert to circle format
        circles = []
        y_max = 5508  # Full height of the image
        x_max = 10908  # Full width of the image
        
        for i, feature in enumerate(top_5_features):
            geometry = feature['geometry']
            if not geometry or not geometry.get('coordinates'):
                continue
            
            # Calculate centroid
            all_coords = []
            if geometry['type'] == "Polygon":
                all_coords = geometry['coordinates']
            elif geometry['type'] == "MultiPolygon":
                all_coords = [coord for polygon in geometry['coordinates'] for coord in polygon]
            
            if not all_coords:
                continue
            
            flat_coords = [coord for ring in all_coords for coord in ring]
            if len(flat_coords) == 0:
                continue
            
            centroid = [
                sum(coord[0] for coord in flat_coords) / len(flat_coords),
                sum(coord[1] for coord in flat_coords) / len(flat_coords)
            ]
            
            # Convert ROI coordinates to Vitessce image coordinates
            # ROI coordinates are in the original image space (10908 x 5508)
            # We need to map them to the Vitessce viewport
            x = centroid[0]  # Keep original X coordinate
            y = centroid[1]  # Keep original Y coordinate from GeoJSON
            
            # Use original coordinates without scaling for better alignment with Vitessce
            x_scaled = x
            y_scaled = y
            
            logger.info(f"ROI {i}: Original centroid: {centroid}, Vitessce coords: x={x}, y={y}")
            
            circle = {
                "id": f"roi_{i}",
                "x": x_scaled,
                "y": y_scaled,
                "original_x": centroid[0],  # Original X from GeoJSON
                "original_y": centroid[1],  # Original Y from GeoJSON
                "score": feature.get('properties', {}).get('score', 0),
                "interactions": feature.get('properties', {}).get('interactions', []),
                "color": get_circle_color(i),
                "radius": 20  # Smaller radius for better visibility
            }
            circles.append(circle)
        
        return jsonify({
            "success": True,
            "circles": circles,
            "total_found": len(filtered_features),
            "showing": len(circles),
            "image_dimensions": {"width": x_max, "height": y_max},
            "viewport_dimensions": {"width": x_max, "height": y_max},
            "scale_factor": 1.0
        })
        
    except Exception as e:
        logger.error(f"Error filtering ROIs: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

def get_circle_color(index):
    """Get color for circle based on index"""
    colors = ['#ff0000', '#00ff00', '#0000ff', '#ffff00', '#ff00ff']
    return colors[index % len(colors)]

################################################################################3

if __name__ == '__main__':
    logger.info("Starting Flask backend server on http://127.0.0.1:5000")
    # Port 5000 is common for backend APIs
    # host='0.0.0.0' makes it accessible from other containers/machines if needed
    app.run(host='127.0.0.1', port=5000, debug=False) 