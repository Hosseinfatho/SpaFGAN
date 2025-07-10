# Backend server code
import json
import logging
import subprocess
from flask import Flask, jsonify, request, send_file
from flask_cors import CORS

import zarr as pyzarr
import s3fs
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

@app.route('/api/channel_names')
def get_channel_names():
    """Reads OME-Zarr metadata to get channel names."""
    logger.info("Request received for /api/channel_names")
    channel_map = {}
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

# === STANDARD VITESSCE CONFIG GENERATOR ===

@app.route('/api/config', methods=['GET'])
def get_config():
    """Get the standard Vitessce config using JavaScript generator"""
    logger.info("Request received for /api/config [GET]")
    
    try:
        # Run the JavaScript config generator
        result = subprocess.run([
            'node', 'generate_config.js'
        ], capture_output=True, text=True, cwd=Path(__file__).parent)
        
        if result.returncode != 0:
            logger.error(f"JavaScript config generation failed: {result.stderr}")
            return jsonify({"error": "Config generation failed"}), 500
        
        config = json.loads(result.stdout)
        return jsonify(config)
        
    except Exception as e:
        logger.error(f"Error generating config: {e}", exc_info=True)
        return jsonify({"error": f"Failed to generate config: {e}"}), 500

@app.route('/api/roi_rectangles_annotation.json', methods=['GET'])
def get_roi_rectangles_annotation():
    """Serve the ROI rectangles annotation JSON file"""
    logger.info("Request received for /api/roi_rectangles_annotation.json [GET]")
    
    try:
        roi_file_path = Path(__file__).parent / 'output' / 'roi_rectangles_annotation.json'
        
        if not roi_file_path.exists():
            logger.error(f"ROI rectangles annotation file not found: {roi_file_path}")
            return jsonify({"error": "ROI rectangles annotation file not found"}), 404
        
        with open(roi_file_path, 'r') as f:
            roi_data = json.load(f)
        
        return jsonify(roi_data)
        
    except Exception as e:
        logger.error(f"Error serving ROI rectangles annotation: {e}", exc_info=True)
        return jsonify({"error": f"Failed to serve ROI rectangles annotation: {e}"}), 500

@app.route('/api/roi_shapes', methods=['GET'])
def serve_roi_shapes():
    try:
        roi_path = Path(__file__).parent / "output" / "roi_rectangles.geojson"
        if not roi_path.exists():
            logger.warning("ROI shapes file not found.")
            return jsonify({"error": "ROI shapes not found"}), 404

        with open(roi_path, 'r') as f:
            roi_data = json.load(f)
        
        # Transform the data to match frontend expectations
        if 'features' in roi_data:
            for feature in roi_data['features']:
                if 'properties' in feature:
                    # Convert 'interaction' string to 'interactions' array
                    if 'interaction' in feature['properties']:
                        feature['properties']['interactions'] = [feature['properties']['interaction']]
                        # Keep the original 'interaction' field for backward compatibility
                    
                    # Ensure 'name' field exists
                    if 'name' not in feature['properties'] and 'id' in feature['properties']:
                        feature['properties']['name'] = feature['properties']['id']
            
        return jsonify(roi_data)
    except Exception as e:
        logger.error(f"Error serving ROI shapes: {e}", exc_info=True)
        return jsonify({"error": f"Failed to serve ROI shapes: {e}"}), 500

@app.route('/api/roi_rectangles_annotation', methods=['GET'])
def serve_roi_rectangles_annotation():
    try:
        roi_path = Path(__file__).parent / "output" / "roi_rectangles_annotation.json"
        if not roi_path.exists():
            logger.warning("ROI rectangles annotation file not found.")
            return jsonify({"error": "ROI rectangles annotation not found"}), 404

        with open(roi_path, 'r') as f:
            roi_data = json.load(f)

        return jsonify(roi_data)
    except Exception as e:
        logger.error(f"Error serving ROI rectangles annotation: {e}")
        return jsonify({"error": "Internal server error"}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000) 