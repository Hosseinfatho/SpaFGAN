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

def generate_vitessce_config():
    """Generate Vitessce configuration in Python"""
    
    # Channel configuration
    channels = [
        {'id': 0, 'name': "CD31", 'color': [0, 255, 0], 'window': [300, 20000], 'targetC': 19},      # Green
        {'id': 1, 'name': "CD20", 'color': [255, 255, 0], 'window': [1000, 7000], 'targetC': 27},    # Yellow
        {'id': 2, 'name': "CD11b", 'color': [255, 0, 255], 'window': [700, 6000], 'targetC': 37},    # Magenta
        {'id': 3, 'name': "CD4", 'color': [0, 255, 255], 'window': [1638, 10000], 'targetC': 25},    # Cyan
        {'id': 4, 'name': "CD11c", 'color': [128, 0, 128], 'window': [370, 1432], 'targetC': 42}     # Purple
    ]

    # Build coordination space
    coordination_space = {
        'dataset': {"A": "bv"},
        'imageChannel': {},
        'imageLayer': {"init_bv_image_0": "__dummy__"},
        'metaCoordinationScopes': {
            "A": {'obsType': "A"},
            "init_bv_image_0": {
                'imageLayer': ["init_bv_image_0"],
                'spatialRenderingMode': "init_bv_image_0",
                'spatialTargetT': "init_bv_image_0",
                'spatialTargetX': "init_bv_image_0",
                'spatialTargetY': "init_bv_image_0",
                'spatialTargetZ': "init_bv_image_0",
                'spatialZoom': "init_bv_image_0"
            }
        },
        'metaCoordinationScopesBy': {
            "A": {},
            "init_bv_image_0": {
                'imageChannel': {
                    'spatialChannelColor': {},
                    'spatialChannelOpacity': {},
                    'spatialChannelVisible': {},
                    'spatialChannelWindow': {},
                    'spatialTargetC': {}
                },
                'imageLayer': {
                    'imageChannel': {"init_bv_image_0": []},
                    'photometricInterpretation': {"init_bv_image_0": "init_bv_image_0"},
                    'spatialLayerOpacity': {"init_bv_image_0": "init_bv_image_0"},
                    'spatialLayerVisible': {"init_bv_image_0": "init_bv_image_0"},
                    'spatialTargetResolution': {"init_bv_image_0": "init_bv_image_0"}
                }
            }
        },
        'obsType': {
            "A": "ROI_B-cell",
            "B": "ROI_Inflammatory", 
            "C": "ROI_T-cell",
            "D": "ROI_Oxidative"
        },
        'obsColorEncoding': {
            "A": "cellSetSelection",
            "B": "cellSetSelection",
            "C": "cellSetSelection",
            "D": "cellSetSelection"
        },
        'photometricInterpretation': {"init_bv_image_0": "BlackIsZero"},
        'spatialChannelColor': {},
        'spatialChannelOpacity': {},
        'spatialChannelVisible': {},
        'spatialChannelWindow': {},
        'spatialLayerOpacity': {"init_bv_image_0": 1.0},
        'spatialLayerVisible': {"init_bv_image_0": True},
        'spatialRenderingMode': {"init_bv_image_0": "3D"},
        'spatialTargetC': {},
        'spatialTargetResolution': {"init_bv_image_0": 3},
        'spatialTargetT': {"init_bv_image_0": 0},
        'spatialTargetX': {"init_bv_image_0": 5454},
        'spatialTargetY': {"init_bv_image_0": 2754},
        'spatialTargetZ': {"init_bv_image_0": 0},
        'spatialZoom': {"init_bv_image_0": -3.0},
        'spatialSegmentationLayer': {
            "A": {
                "radius": 500,
                "stroked": True,
                "visible": True,
                "opacity": 0.8,
                "color": [255, 100, 100]  # Red for B-cell infiltration
            },
            "B": {
                "radius": 500,
                "stroked": True,
                "visible": True,
                "opacity": 0.8,
                "color": [0, 255, 0]  # Green for Inflammatory zone
            },
            "C": {
                "radius": 500,
                "stroked": True,
                "visible": True,
                "opacity": 0.8,
                "color": [0, 0, 255]  # Blue for T-cell entry site
            },
            "D": {
                "radius": 500,
                "stroked": True,
                "visible": True,
                "opacity": 0.8,
                "color": [255, 255, 0]  # Yellow for Oxidative stress niche
            }
        }
    }

    # Add channel-specific coordination values
    for i, channel in enumerate(channels):
        ch_id = f'init_bv_image_{i}'
        coordination_space['imageChannel'][ch_id] = "__dummy__"
        coordination_space['spatialChannelColor'][ch_id] = channel['color']
        coordination_space['spatialChannelOpacity'][ch_id] = 1.0
        coordination_space['spatialChannelVisible'][ch_id] = True
        coordination_space['spatialChannelWindow'][ch_id] = channel['window']
        coordination_space['spatialTargetC'][ch_id] = channel['targetC']
        
        # Add to meta coordination scopes
        for key in ['spatialChannelColor', 'spatialChannelOpacity', 'spatialChannelVisible', 'spatialChannelWindow', 'spatialTargetC']:
            coordination_space['metaCoordinationScopesBy']['init_bv_image_0']['imageChannel'][key][ch_id] = ch_id

    config = {
        'version': '1.0.16',
        'name': 'BioMedVis Challenge',
        'description': 'ROI annotations for the BioMedVis Challenge',
        'datasets': [{
            'uid': 'bv',
            'name': 'Blood Vessel',
            'files': [
                {
                    'fileType': 'image.ome-zarr',
                    'url': 'https://lsp-public-data.s3.amazonaws.com/yapp-2023-3d-melanoma/Dataset1-LSP13626-melanoma-in-situ/0',
                },
                {
                    'fileType': 'obsSegmentations.json',
                    'url': 'http://localhost:5000/api/roi_segmentation_B-cell_infiltration.json',
                    'coordinationValues': {
                        'obsType': 'ROI_B-cell',
                    },
                },
                {
                    'fileType': 'obsSegmentations.json',
                    'url': 'http://localhost:5000/api/roi_segmentation_Inflammatory_zone.json',
                    'coordinationValues': {
                        'obsType': 'ROI_Inflammatory',
                    },
                },
                {
                    'fileType': 'obsSegmentations.json',
                    'url': 'http://localhost:5000/api/roi_segmentation_T-cell_entry_site.json',
                    'coordinationValues': {
                        'obsType': 'ROI_T-cell',
                    },
                },
                {
                    'fileType': 'obsSegmentations.json',
                    'url': 'http://localhost:5000/api/roi_segmentation_Oxidative_stress_niche.json',
                    'coordinationValues': {
                        'obsType': 'ROI_Oxidative',
                    },
                }
            ]
        }],
        'initStrategy': 'auto',
        'coordinationSpace': coordination_space,
        'layout': [
            {
                'component': 'spatialBeta',
                'coordinationScopes': {
                    'dataset': "A",
                    'metaCoordinationScopes': ["init_bv_image_0", "A"],
                    'metaCoordinationScopesBy': ["init_bv_image_0", "A"],
                    'spatialSegmentationLayer': ["A", "B", "C", "D"],
                    'obsType': ["A", "B", "C", "D"],
                    'obsColorEncoding': ["A", "B", "C", "D"]
                },
                'x': 0, 'y': 0, 'w': 6, 'h': 12
            },
            {
                'component': 'layerControllerBeta',
                'coordinationScopes': {
                    'dataset': "A",
                    'metaCoordinationScopes': ["init_bv_image_0", "A"],
                    'metaCoordinationScopesBy': ["init_bv_image_0", "A"],
                    'spatialSegmentationLayer': ["A", "B", "C", "D"],
                    'obsType': ["A", "B", "C", "D"]
                },
                'x': 6, 'y': 0, 'w': 6, 'h': 12
            }
        ]
    }

    return config

@app.route('/api/config', methods=['GET'])
def get_config():
    """Get the standard Vitessce config using Python generator"""
    logger.info("Request received for /api/config [GET]")
    
    try:
        config = generate_vitessce_config()
        return jsonify(config)
        
    except Exception as e:
        logger.error(f"Error generating config: {e}", exc_info=True)
        return jsonify({"error": f"Failed to generate config: {e}"}), 500

@app.route('/api/roi_segmentation_B-cell_infiltration.json', methods=['GET'])
def get_roi_segmentation_b_cell():
    """Serve the B-cell infiltration ROI segmentation JSON file"""
    logger.info("Request received for /api/roi_segmentation_B-cell_infiltration.json [GET]")
    
    try:
        roi_file_path = Path(__file__).parent / 'output' / 'roi_segmentation_B-cell_infiltration.json'
        
        if not roi_file_path.exists():
            logger.error(f"B-cell infiltration ROI segmentation file not found: {roi_file_path}")
            return jsonify({"error": "B-cell infiltration ROI segmentation file not found"}), 404
        
        with open(roi_file_path, 'r') as f:
            roi_data = json.load(f)
        
        return jsonify(roi_data)
        
    except Exception as e:
        logger.error(f"Error serving B-cell infiltration ROI segmentation: {e}", exc_info=True)
        return jsonify({"error": f"Failed to serve B-cell infiltration ROI segmentation: {e}"}), 500

@app.route('/api/roi_segmentation_T-cell_entry_site.json', methods=['GET'])
def get_roi_segmentation_t_cell():
    """Serve the T-cell entry site ROI segmentation JSON file"""
    logger.info("Request received for /api/roi_segmentation_T-cell_entry_site.json [GET]")
    
    try:
        roi_file_path = Path(__file__).parent / 'output' / 'roi_segmentation_T-cell_entry_site.json'
        
        if not roi_file_path.exists():
            logger.error(f"T-cell entry site ROI segmentation file not found: {roi_file_path}")
            return jsonify({"error": "T-cell entry site ROI segmentation file not found"}), 404
        
        with open(roi_file_path, 'r') as f:
            roi_data = json.load(f)
        
        return jsonify(roi_data)
        
    except Exception as e:
        logger.error(f"Error serving T-cell entry site ROI segmentation: {e}", exc_info=True)
        return jsonify({"error": f"Failed to serve T-cell entry site ROI segmentation: {e}"}), 500

@app.route('/api/roi_segmentation_Inflammatory_zone.json', methods=['GET'])
def get_roi_segmentation_inflammatory():
    """Serve the Inflammatory zone ROI segmentation JSON file"""
    logger.info("Request received for /api/roi_segmentation_Inflammatory_zone.json [GET]")
    
    try:
        roi_file_path = Path(__file__).parent / 'output' / 'roi_segmentation_Inflammatory_zone.json'
        
        if not roi_file_path.exists():
            logger.error(f"Inflammatory zone ROI segmentation file not found: {roi_file_path}")
            return jsonify({"error": "Inflammatory zone ROI segmentation file not found"}), 404
        
        with open(roi_file_path, 'r') as f:
            roi_data = json.load(f)
        
        return jsonify(roi_data)
        
    except Exception as e:
        logger.error(f"Error serving Inflammatory zone ROI segmentation: {e}", exc_info=True)
        return jsonify({"error": f"Failed to serve Inflammatory zone ROI segmentation: {e}"}), 500

@app.route('/api/roi_segmentation_Oxidative_stress_niche.json', methods=['GET'])
def get_roi_segmentation_oxidative():
    """Serve the Oxidative stress niche ROI segmentation JSON file"""
    logger.info("Request received for /api/roi_segmentation_Oxidative_stress_niche.json [GET]")
    
    try:
        roi_file_path = Path(__file__).parent / 'output' / 'roi_segmentation_Oxidative_stress_niche.json'
        
        if not roi_file_path.exists():
            logger.error(f"Oxidative stress niche ROI segmentation file not found: {roi_file_path}")
            return jsonify({"error": "Oxidative stress niche ROI segmentation file not found"}), 404
        
        with open(roi_file_path, 'r') as f:
            roi_data = json.load(f)
        
        return jsonify(roi_data)
        
    except Exception as e:
        logger.error(f"Error serving Oxidative stress niche ROI segmentation: {e}", exc_info=True)
        return jsonify({"error": f"Failed to serve Oxidative stress niche ROI segmentation: {e}"}), 500

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