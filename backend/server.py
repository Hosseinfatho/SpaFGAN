# Backend server code
import json
import logging
from flask import Flask, jsonify, request, Response, stream_with_context
from flask_cors import CORS
import requests
from vitessce import (
    VitessceConfig,
    CoordinationLevel as CL,
    get_initial_coordination_scope_prefix
)
import zarr as pyzarr
import numpy as np
from scipy import ndimage
import traceback
import s3fs
import time
import os
from pathlib import Path

# Configure basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Constants for Zarr Access
ZARR_BASE_URL = "s3://lsp-public-data/yapp-2023-3d-melanoma/Dataset1-LSP13626-melanoma-in-situ"
ZARR_IMAGE_GROUP_PATH = "0"
TARGET_RESOLUTION_PATH = "3"

# Channel indices for RGB interactions
REQUIRED_CHANNEL_INDICES_FOR_RGB_INTERACTION = {
    'CD31': 19,
    'CD11b': 37,
    'Catalase': 59,
    'CD4': 25,
    'CD20': 27,
    'CD11c': 41
}

# RGB interaction definitions
RGB_INTERACTION_DEFINITIONS = [
    {'name': 'Vascular inflammation + oxidative stress', 'channels': ['CD31', 'CD11b', 'Catalase'], 'rgb_target': 0},
    {'name': 'Complex immune cell interaction', 'channels': ['CD4', 'CD20', 'CD11b'], 'rgb_target': 2},
    {'name': 'T/B cell entry via vessels', 'channels': ['CD31', 'CD4', 'CD20'], 'rgb_target': 1}
]

@app.route('/api/channel_names')
def get_channel_names():
    """Reads OME-Zarr metadata to get channel names."""
    logger.info("Request received for /api/channel_names")
    channel_map = {}
    s3_local = None
    root_store = None
    root_group = None
    try:
        logger.info("Attempting to create S3FileSystem...")
        s3_local = s3fs.S3FileSystem(anon=True)
        logger.info("S3FileSystem created successfully.")
        
        logger.info(f"Attempting to create S3Map for root: {ZARR_BASE_URL}")
        root_store = s3fs.S3Map(root=ZARR_BASE_URL, s3=s3_local, check=False)
        logger.info("S3Map created successfully.")

        logger.info("Attempting to open root Zarr group...")
        root_group = pyzarr.open_consolidated(store=root_store) if '.zmetadata' in root_store else pyzarr.open_group(store=root_store, mode='r')
        logger.info(f"Root group opened successfully. Type: {type(root_group)}")

        # Access OME metadata: Look inside the image group (e.g., '0')
        omero_meta = None
        image_group_key = '0'
        logger.info(f"Checking for image group '{image_group_key}'...")
        if root_group and image_group_key in root_group:
            image_group = root_group[image_group_key]
            logger.info(f"Image group '{image_group_key}' found. Checking for 'omero' in its attributes...")
            if hasattr(image_group, 'attrs') and 'omero' in image_group.attrs:
                omero_meta = image_group.attrs['omero']
                logger.info(f"Found 'omero' metadata in image group '{image_group_key}' attributes.")
            else:
                 logger.warning(f"Could not find 'omero' metadata in image group '{image_group_key}' attributes.")
        else:
             logger.warning(f"Image group '{image_group_key}' not found in root group.")

        # Now check the extracted omero_meta for channels
        if omero_meta and 'channels' in omero_meta:
            logger.info(f"Found {len(omero_meta['channels'])} channels in metadata.")
            for i, channel_info in enumerate(omero_meta['channels']): 
                channel_map[str(i)] = channel_info.get('label', f'Channel {i}') 
            logger.info(f" Successfully extracted channel names: {channel_map}")
            return jsonify(channel_map)
        else:
            logger.warning("Could not find valid 'omero' metadata with 'channels'. Returning 404.")
            return jsonify({"error": "Channel metadata ('omero' with 'channels') not found in Zarr store"}), 404

    except Exception as e:
        logger.error(f" Failed during channel name retrieval: {e}", exc_info=True)
        logger.error(f"State at error: s3_local={'Exists' if s3_local else 'None'}, root_store={'Exists' if root_store else 'None'}, root_group={'Exists' if root_group else 'None'}")
        return jsonify({"error": f"Failed to read channel names: {e}"}), 500

def open_target_zarr_array():
    """Opens and returns the target Zarr array (/0/3). Returns None on failure."""
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
        circle_file = output_dir / "interactive_roi_circles.geojson"
        logger.info(f"Checking for circle file: {circle_file}")
        logger.info(f"Circle file exists: {circle_file.exists()}")
        if circle_file.exists():
            logger.info("Adding circle layer to dataset")
            # dataset.add_file(
            #     url="http://localhost:5000/api/roi_circles",
            #     file_type="obsSegmentations.json"
            # )

        # Add ROI annotations as GeoJSON layer
        roi_file = output_dir / "roi_shapes.geojson"
        logger.info(f"Checking for ROI file: {roi_file}")
        logger.info(f"ROI file exists: {roi_file.exists()}")
        if roi_file.exists():
            logger.info("Adding ROI annotations layer to dataset")
            # dataset.add_file(
            #     url="http://localhost:5000/api/roi_shapes",
            #     file_type="obsSegmentations.json"
            # )

        # Add rectangles annotation as GeoJSON layer
        rectangle_file = output_dir / "roi_rectangles_annotation.json"
        logger.info(f"Checking for rectangles annotation file: {rectangle_file}")
        logger.info(f"Rectangles annotation file exists: {rectangle_file.exists()}")
        if rectangle_file.exists():
            logger.info("Adding rectangles annotation layer to dataset")
            dataset.add_file(
                url="http://localhost:5000/api/roi_rectangles_annotation",
                file_type="obsSegmentations.json"
            )

        spatial = vc.add_view("spatialBeta", dataset=dataset)
        lc = vc.add_view("layerControllerBeta", dataset=dataset)

        # Reconstruct CL objects from view_state_data
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
        # Force the segmentation layer name to "ROI"
        for dataset in config_dict.get("datasets", []):
            for file in dataset.get("files", []):
                if file.get("fileType") == "obsSegmentations.json":
                    file["options"] = {"name": "ROI"}
        # Example: Remove options from all files
        for dataset in config_dict.get("datasets", []):
            for file in dataset.get("files", []):
                if "options" in file:
                    del file["options"]
        logger.info(f" Vitessce Configuration generated successfully from POST data.")
        return config_dict
    except Exception as e:
        logger.error(f"Error generating Vitessce config from POST data: {e}", exc_info=True)
        return None

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

@app.route('/api/zarr-proxy')
def zarr_proxy():
    """Proxies requests to a Zarr URL to bypass CORS issues."""
    target_url = request.args.get('url')
    if not target_url:
        return jsonify({"error": "Missing 'url' query parameter"}), 400

    logger.info(f"Proxying request for Zarr URL: {target_url}")

    try:
        resp = requests.get(target_url, stream=True, headers=request.headers)
        resp.raise_for_status()

        response_headers = {
            'Content-Type': resp.headers.get('Content-Type', 'application/octet-stream'),
        }

        return Response(stream_with_context(resp.iter_content(chunk_size=8192)),
                        status=resp.status_code,
                        headers=response_headers)

    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching Zarr URL {target_url}: {e}")
        return jsonify({"error": f"Failed to fetch Zarr URL: {e}"}), 502
    except Exception as e:
        logger.error(f"Unexpected error in Zarr proxy: {e}", exc_info=True)
        return jsonify({"error": "Internal server error in proxy"}), 500

@app.route('/api/roi_shapes', methods=['GET'])
def serve_roi_shapes():
    try:
        roi_path = Path(__file__).parent / "output" / "roi_shapes.geojson"
        if not roi_path.exists():
            logger.warning("ROI shapes file not found.")
            return jsonify({"error": "ROI shapes not found"}), 404

        with open(roi_path, 'r') as f:
                roi_data = json.load(f)
            
        return jsonify(roi_data)
    except Exception as e:
        logger.error(f"Error serving ROI shapes: {e}", exc_info=True)
        return jsonify({"error": f"Failed to serve ROI shapes: {e}"}), 500

@app.route('/api/roi_circles', methods=['GET'])
def serve_roi_circles():
    try:
        circle_path = Path(__file__).parent / "output" / "interactive_roi_circles.geojson"
        if not circle_path.exists():
            logger.warning("ROI circles file not found.")
            return jsonify({"error": "ROI circles not found"}), 404
        
        with open(circle_path, 'r') as f:
                circle_data = json.load(f)
            
        return jsonify(circle_data)
    except Exception as e:
        logger.error(f"Error serving ROI circles: {e}", exc_info=True)
        return jsonify({"error": f"Failed to serve ROI circles: {e}"}), 500

@app.route('/api/roi_rectangles_annotation', methods=['GET'])
def serve_roi_rectangles_annotation():
    try:
        annotation_path = Path(__file__).parent / "output" / "roi_rectangles_annotation.json"
        if not annotation_path.exists():
            return jsonify({"error": "ROI rectangles annotation not found"}), 404
        with open(annotation_path, 'r') as f:
            annotation_data = json.load(f)
        return jsonify(annotation_data)
    except Exception as e:
        return jsonify({"error": f"Failed to serve ROI rectangles annotation: {e}"}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000) 