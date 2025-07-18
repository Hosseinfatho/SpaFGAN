

import json
import logging
from flask_cors import CORS
from flask import Flask, jsonify, send_from_directory, request
import os
from collections import OrderedDict

import zarr as pyzarr
import s3fs
from pathlib import Path

ZARR_BASE_DIR = os.path.abspath("output/roi_shapes.spatialdata.zarr")

# Configure basic logging
logging.basicConfig(level=logging.WARNING)  # Reduce logging level for better performance
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Enable Flask caching for better performance
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 3600  # Cache static files for 1 hour



@app.route('/api/channel_names')
def get_channel_names():
    """Reads OME-Zarr metadata to get channel names."""
    logger.info("Request received for /api/channel_names")
    channel_map = {}
    root_store = None
    root_group = None
    try:
        s3_local = s3fs.S3FileSystem(anon=True)
        root_store = s3fs.S3Map(root=ZARR_BASE_URL, s3=s3_local, check=False)
        root_group = pyzarr.open_consolidated(store=root_store) if '.zmetadata' in root_store else pyzarr.open_group(store=root_store, mode='r')
        omero_meta = None
        image_group_key = '0'
        if root_group and image_group_key in root_group:
            image_group = root_group[image_group_key]
            if hasattr(image_group, 'attrs') and 'omero' in image_group.attrs:
                omero_meta = image_group.attrs['omero']
            else:
                 logger.warning(f"Could not find 'omero' metadata in image group '{image_group_key}' attributes.")
        else:
             logger.warning(f"Image group '{image_group_key}' not found in root group.")
        # Now check the extracted omero_meta for channels
        if omero_meta and 'channels' in omero_meta:
            for i, channel_info in enumerate(omero_meta['channels']): 
                channel_map[str(i)] = channel_info.get('label', f'Channel {i}') 
            return jsonify(channel_map)
        else:
            return jsonify({"error": "Channel metadata ('omero' with 'channels') not found in Zarr store"}), 404
    except Exception as e:
        return jsonify({"error": f"Failed to read channel names: {e}"}), 500

# Define absolute path to the .sdata.zarr directory
@app.route("/api/image/<path:filename>", methods=["GET"])
def serve_image_file(filename):
    """Serve image files from the output directory"""
    try:
        image_dir = Path(__file__).parent / "output"
        return send_from_directory(image_dir, filename)
    except Exception as e:
        logger.error(f"Error serving image file '{filename}': {e}", exc_info=True)
        return jsonify({"error": f"Failed to serve image file '{filename}': {e}"}), 500

@app.route('/api/roi_segmentation.json', methods=['GET'])
def get_roi_segmentation():
    """Serve the roi_Segmentation.json file"""
    logger.info("Request received for /api/roi_segmentation.json [GET]")
    
    try:
        roi_file_path = Path(__file__).parent / 'output' / 'roi_Segmentation.json'
        
        if not roi_file_path.exists():
            logger.error(f"roi_Segmentation.json file not found: {roi_file_path}")
            return jsonify({"error": "roi_Segmentation.json file not found"}), 404
        
        with open(roi_file_path, 'r') as f:
            roi_data = json.load(f)
        
        return jsonify(roi_data)
        
    except Exception as e:
        logger.error(f"Error serving roi_Segmentation.json: {e}", exc_info=True)
        return jsonify({"error": f"Failed to serve roi_Segmentation.json: {e}"}), 500

@app.route('/api/roi_segmentation_B-cell_infiltration.json', methods=['GET'])
def get_roi_segmentation_b_cell():
    """Serve the roi_segmentation_B-cell_infiltration.json file"""
    logger.info("Request received for /api/roi_segmentation_B-cell_infiltration.json [GET]")
    
    try:
        roi_file_path = Path(__file__).parent / 'output' / 'roi_segmentation_B-cell_infiltration.json'
        
        if not roi_file_path.exists():
            logger.error(f"roi_segmentation_B-cell_infiltration.json file not found: {roi_file_path}")
            return jsonify({"error": "roi_segmentation_B-cell_infiltration.json file not found"}), 404
        
        with open(roi_file_path, 'r') as f:
            roi_data = json.load(f)
        
        return jsonify(roi_data)
        
    except Exception as e:
        logger.error(f"Error serving roi_segmentation_B-cell_infiltration.json: {e}", exc_info=True)
        return jsonify({"error": f"Failed to serve roi_segmentation_B-cell_infiltration.json: {e}"}), 500

@app.route('/api/roi_segmentation_Inflammatory_zone.json', methods=['GET'])
def get_roi_segmentation_inflammatory():
    """Serve the roi_segmentation_Inflammatory_zone.json file"""
    logger.info("Request received for /api/roi_segmentation_Inflammatory_zone.json [GET]")
    
    try:
        roi_file_path = Path(__file__).parent / 'output' / 'roi_segmentation_Inflammatory_zone.json'
        
        if not roi_file_path.exists():
            logger.error(f"roi_segmentation_Inflammatory_zone.json file not found: {roi_file_path}")
            return jsonify({"error": "roi_segmentation_Inflammatory_zone.json file not found"}), 404
        
        with open(roi_file_path, 'r') as f:
            roi_data = json.load(f)
        
        return jsonify(roi_data)
        
    except Exception as e:
        logger.error(f"Error serving roi_segmentation_Inflammatory_zone.json: {e}", exc_info=True)
        return jsonify({"error": f"Failed to serve roi_segmentation_Inflammatory_zone.json: {e}"}), 500

@app.route('/api/roi_segmentation_T-cell_entry_site.json', methods=['GET'])
def get_roi_segmentation_t_cell():
    """Serve the roi_segmentation_T-cell_entry_site.json file"""
    logger.info("Request received for /api/roi_segmentation_T-cell_entry_site.json [GET]")
    
    try:
        roi_file_path = Path(__file__).parent / 'output' / 'roi_segmentation_T-cell_entry_site.json'
        
        if not roi_file_path.exists():
            logger.error(f"roi_segmentation_T-cell_entry_site.json file not found: {roi_file_path}")
            return jsonify({"error": "roi_segmentation_T-cell_entry_site.json file not found"}), 404
        
        with open(roi_file_path, 'r') as f:
            roi_data = json.load(f)
        
        return jsonify(roi_data)
        
    except Exception as e:
        logger.error(f"Error serving roi_segmentation_T-cell_entry_site.json: {e}", exc_info=True)
        return jsonify({"error": f"Failed to serve roi_segmentation_T-cell_entry_site.json: {e}"}), 500

@app.route('/api/roi_segmentation_Oxidative_stress_niche.json', methods=['GET'])
def get_roi_segmentation_oxidative():
    """Serve the roi_segmentation_Oxidative_stress_niche.json file"""
    logger.info("Request received for /api/roi_segmentation_Oxidative_stress_niche.json [GET]")
    
    try:
        roi_file_path = Path(__file__).parent / 'output' / 'roi_segmentation_Oxidative_stress_niche.json'
        
        if not roi_file_path.exists():
            logger.error(f"roi_segmentation_Oxidative_stress_niche.json file not found: {roi_file_path}")
            return jsonify({"error": "roi_segmentation_Oxidative_stress_niche.json file not found"}), 404
        
        with open(roi_file_path, 'r') as f:
            roi_data = json.load(f)
        
        return jsonify(roi_data)
        
    except Exception as e:
        logger.error(f"Error serving roi_segmentation_Oxidative_stress_niche.json: {e}", exc_info=True)
        return jsonify({"error": f"Failed to serve roi_segmentation_Oxidative_stress_niche.json: {e}"}), 500

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

@app.route("/api/segmentation/<path:filename>", methods=["GET"])
def serve_segmentation_file(filename):
    """Serve segmentation files (ome-tif and offsets) with proper headers"""
    try:
        logger.info(f"Serving segmentation file: {filename}")
        
        # First try vitessce_files directory
        segmentation_dir = Path(__file__).parent / "output" / "vitessce_files"
        file_path = segmentation_dir / filename
        
        if file_path.exists():
            response = send_from_directory(segmentation_dir, filename)
            # Add headers to prevent range request issues
            response.headers['Accept-Ranges'] = 'none'
            response.headers['Cache-Control'] = 'public, max-age=3600'
            return response
        else:
            # If not found in vitessce_files, try output directory
            segmentation_dir = Path(__file__).parent / "output"
            file_path = segmentation_dir / filename
            
            if file_path.exists():
                response = send_from_directory(segmentation_dir, filename)
                # Add headers to prevent range request issues
                response.headers['Accept-Ranges'] = 'none'
                response.headers['Cache-Control'] = 'public, max-age=3600'
                return response
            else:
                logger.error(f"Segmentation file not found: {filename}")
                return jsonify({"error": f"Segmentation file not found: {filename}"}), 404
                
    except Exception as e:
        logger.error(f"Error serving segmentation file '{filename}': {e}", exc_info=True)
        return jsonify({"error": f"Failed to serve segmentation file '{filename}': {e}"}), 500

# Dynamic config generation based on selected interaction types


# Config generation moved to frontend - backend config API removed
# Backend now only serves ROI files and other static data

@app.route('/api/updateconfig', methods=['GET', 'POST'])
def get_current_config():
    """Get or update the current config"""
    logger.info(f"Request received for /api/updateconfig [{request.method}]")
    
    try:
        if request.method == 'POST':
            # Receive config from frontend
            config_data = request.get_json()
            if config_data:
                # Store the config in memory
                app.config['current_vitessce_config'] = config_data
                logger.info("Config updated from frontend")
                return jsonify({"message": "Config updated successfully", "status": "success"})
            else:
                return jsonify({"error": "No config data received"}), 400
        
        elif request.method == 'GET':
            # Return the stored config
            stored_config = app.config.get('current_vitessce_config')
            if stored_config:
                return jsonify(stored_config)
            else:
                return jsonify({
                    "message": "No config available yet",
                    "instructions": [
                        "1. Open frontend at http://localhost:5173",
                        "2. Select some ROI groups",
                        "3. Config will be automatically sent to this endpoint",
                        "4. Refresh this page to see the updated config"
                    ],
                    "status": "waiting_for_config"
                })
        
    except Exception as e:
        logger.error(f"Error in updateconfig endpoint: {e}", exc_info=True)
        return jsonify({"error": f"Failed to handle config: {e}"}), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000) 
    