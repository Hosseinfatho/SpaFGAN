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
    """Generate Vitessce configuration based on Maynard et al. example"""
    
    # Build coordination space
    coordination_space = {
        'dataset': {"A": "A"},
        'featureSelection': {"A": ["33334"]},
        'obsColorEncoding': {"A": "geneSelection"},
        'spatialChannelVisible': {"A": True},
        'spatialChannelOpacity': {"A": 0.5},
        'spotLayer': {"init_A_obsSpots_0": "__dummy__"},
        'imageLayer': {"init_A_image_0": "__dummy__"},
        'photometricInterpretation': {"init_A_image_0": "RGB"},
        'segmentationLayer': {"init_A_obsSegmentations_0": "__dummy__"},
        'segmentationChannel': {
            "init_A_obsSegmentations_0": "__dummy__",
            "init_A_obsSegmentations_1": "__dummy__",
            "init_A_obsSegmentations_2": "__dummy__",
            "init_A_obsSegmentations_3": "__dummy__",
            "init_A_obsSegmentations_4": "__dummy__",
            "init_A_obsSegmentations_5": "__dummy__",
            "init_A_obsSegmentations_6": "__dummy__"
        },
        'obsType': {
            "init_A_obsSegmentations_0": "White Matter",
            "init_A_obsSegmentations_1": "Layer 6",
            "init_A_obsSegmentations_2": "Layer 5",
            "init_A_obsSegmentations_3": "Layer 4",
            "init_A_obsSegmentations_4": "Layer 3",
            "init_A_obsSegmentations_5": "Layer 2",
            "init_A_obsSegmentations_6": "Layer 1",
            "A": "spot"
        },
        'spatialTargetC': {
            "init_A_obsSegmentations_0": 6,
            "init_A_obsSegmentations_1": 5,
            "init_A_obsSegmentations_2": 4,
            "init_A_obsSegmentations_3": 3,
            "init_A_obsSegmentations_4": 2,
            "init_A_obsSegmentations_5": 1,
            "init_A_obsSegmentations_6": 0
        },
        'spatialChannelColor': {
            "init_A_obsSegmentations_0": [202, 104, 119],
            "init_A_obsSegmentations_1": [221, 204, 128],
            "init_A_obsSegmentations_2": [153, 153, 65],
            "init_A_obsSegmentations_3": [28, 118, 58],
            "init_A_obsSegmentations_4": [74, 169, 154],
            "init_A_obsSegmentations_5": [139, 203, 235],
            "init_A_obsSegmentations_6": [70, 119, 167]
        },
        'metaCoordinationScopes': {
            "init_A_obsSpots_0": {
                'spotLayer': "init_A_obsSpots_0"
            },
            "init_A_image_0": {
                'imageLayer': "init_A_image_0"
            },
            "init_A_obsSegmentations_0": {
                'segmentationLayer': ["init_A_obsSegmentations_0"]
            }
        },
        'metaCoordinationScopesBy': {
            "init_A_obsSpots_0": {
                'spotLayer': {
                    'obsColorEncoding': {"init_A_obsSpots_0": "A"},
                    'featureSelection': {"init_A_obsSpots_0": "A"}
                }
            },
            "init_A_image_0": {
                'imageLayer': {
                    'photometricInterpretation': {"init_A_image_0": "init_A_image_0"}
                }
            },
            "init_A_obsSegmentations_0": {
                'segmentationLayer': {
                    'segmentationChannel': {
                        "init_A_obsSegmentations_0": [
                            "init_A_obsSegmentations_0",
                            "init_A_obsSegmentations_1",
                            "init_A_obsSegmentations_2",
                            "init_A_obsSegmentations_3",
                            "init_A_obsSegmentations_4",
                            "init_A_obsSegmentations_5",
                            "init_A_obsSegmentations_6"
                        ]
                    }
                },
                'segmentationChannel': {
                    'obsType': {
                        "init_A_obsSegmentations_0": "init_A_obsSegmentations_0",
                        "init_A_obsSegmentations_1": "init_A_obsSegmentations_1",
                        "init_A_obsSegmentations_2": "init_A_obsSegmentations_2",
                        "init_A_obsSegmentations_3": "init_A_obsSegmentations_3",
                        "init_A_obsSegmentations_4": "init_A_obsSegmentations_4",
                        "init_A_obsSegmentations_5": "init_A_obsSegmentations_5",
                        "init_A_obsSegmentations_6": "init_A_obsSegmentations_6"
                    },
                    'spatialTargetC': {
                        "init_A_obsSegmentations_0": "init_A_obsSegmentations_0",
                        "init_A_obsSegmentations_1": "init_A_obsSegmentations_1",
                        "init_A_obsSegmentations_2": "init_A_obsSegmentations_2",
                        "init_A_obsSegmentations_3": "init_A_obsSegmentations_3",
                        "init_A_obsSegmentations_4": "init_A_obsSegmentations_4",
                        "init_A_obsSegmentations_5": "init_A_obsSegmentations_5",
                        "init_A_obsSegmentations_6": "init_A_obsSegmentations_6"
                    },
                    'spatialChannelVisible': {
                        "init_A_obsSegmentations_0": "A",
                        "init_A_obsSegmentations_1": "A",
                        "init_A_obsSegmentations_2": "A",
                        "init_A_obsSegmentations_3": "A",
                        "init_A_obsSegmentations_4": "A",
                        "init_A_obsSegmentations_5": "A",
                        "init_A_obsSegmentations_6": "A"
                    },
                    'spatialChannelOpacity': {
                        "init_A_obsSegmentations_0": "A",
                        "init_A_obsSegmentations_1": "A",
                        "init_A_obsSegmentations_2": "A",
                        "init_A_obsSegmentations_3": "A",
                        "init_A_obsSegmentations_4": "A",
                        "init_A_obsSegmentations_5": "A",
                        "init_A_obsSegmentations_6": "A"
                    },
                    'spatialChannelColor': {
                        "init_A_obsSegmentations_0": "init_A_obsSegmentations_0",
                        "init_A_obsSegmentations_1": "init_A_obsSegmentations_1",
                        "init_A_obsSegmentations_2": "init_A_obsSegmentations_2",
                        "init_A_obsSegmentations_3": "init_A_obsSegmentations_3",
                        "init_A_obsSegmentations_4": "init_A_obsSegmentations_4",
                        "init_A_obsSegmentations_5": "init_A_obsSegmentations_5",
                        "init_A_obsSegmentations_6": "init_A_obsSegmentations_6"
                    }
                }
            }
        }
    }

    config = {
        'version': '1.0.16',
        'name': 'Maynard et al., Nature Neuroscience 2021',
        'description': 'Human dorsolateral prefrontal cortex profiled by 10x Genomics Visium',
        'datasets': [{
            'uid': 'A',
            'name': 'My dataset',
            'files': [
                {
                    'url': 'https://storage.googleapis.com/vitessce-demo-data/maynard-2021/151673.sdata.zarr',
                    'fileType': 'image.spatialdata.zarr',
                    'options': {
                        'path': 'images/visium_151673_full_image'
                    }
                },
                {
                    'url': 'https://storage.googleapis.com/vitessce-demo-data/maynard-2021/151673.sdata.zarr',
                    'fileType': 'labels.spatialdata.zarr',
                    'options': {
                        'path': 'images/visium_151673_annotations'
                    }
                },
                {
                    'url': 'https://storage.googleapis.com/vitessce-demo-data/maynard-2021/151673.sdata.zarr',
                    'fileType': 'obsFeatureMatrix.spatialdata.zarr',
                    'coordinationValues': {
                        'obsType': 'spot'
                    },
                    'options': {
                        'path': 'table/table/layers/logcounts',
                        'initialFeatureFilterPath': 'table/table/var/is_top_hvg'
                    }
                },
                {
                    'url': 'https://storage.googleapis.com/vitessce-demo-data/maynard-2021/151673.sdata.zarr',
                    'fileType': 'featureLabels.spatialdata.zarr',
                    'options': {
                        'path': 'table/table/var/gene_name'
                    }
                },
                {
                    'url': 'https://storage.googleapis.com/vitessce-demo-data/maynard-2021/151673.sdata.zarr',
                    'fileType': 'obsSpots.spatialdata.zarr',
                    'coordinationValues': {
                        'obsType': 'spot'
                    },
                    'options': {
                        'path': 'shapes/visium_151673',
                        'tablePath': 'table/table'
                    }
                },
                {
                    'url': 'https://storage.googleapis.com/vitessce-demo-data/maynard-2021/151673.sdata.zarr',
                    'fileType': 'obsSets.spatialdata.zarr',
                    'coordinationValues': {
                        'obsType': 'spot'
                    },
                    'options': {
                        'obsSets': [
                            {
                                'name': 'Layer',
                                'path': 'table/table/obs/layer_manual'
                            }
                        ]
                    }
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
                    'metaCoordinationScopes': [
                        "init_A_obsSpots_0",
                        "init_A_image_0",
                        "init_A_obsSegmentations_0"
                    ],
                    'metaCoordinationScopesBy': [
                        "init_A_obsSpots_0",
                        "init_A_image_0",
                        "init_A_obsSegmentations_0"
                    ],
                    'obsType': "A"
                },
                'x': 0, 'y': 0, 'w': 6, 'h': 12
            },
            {
                'component': 'layerControllerBeta',
                'coordinationScopes': {
                    'dataset': "A",
                    'metaCoordinationScopes': [
                        "init_A_obsSpots_0",
                        "init_A_image_0",
                        "init_A_obsSegmentations_0"
                    ],
                    'metaCoordinationScopesBy': [
                        "init_A_obsSpots_0",
                        "init_A_image_0",
                        "init_A_obsSegmentations_0"
                    ],
                    'obsType': "A"
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

@app.route('/api/spatialdata_zarr', methods=['GET'])
def serve_spatialdata_zarr():
    """Serve SpatialData Zarr files"""
    try:
        # Get the path parameter from the request
        path_param = request.args.get('path', '')
        logger.info(f"Request for SpatialData Zarr with path: {path_param}")
        
        # Determine which Zarr file to serve based on the path
        if 'shapes' in path_param or 'roi_circles' in path_param:
            zarr_path = Path(__file__).parent / "output" / "roi_shapes.spatialdata.zarr"
        elif 'table' in path_param:
            zarr_path = Path(__file__).parent / "output" / "roi_shapes.spatialdata.zarr"
        elif 'images' in path_param:
            zarr_path = Path(__file__).parent / "output" / "test.zarr"
        else:
            # Default to roi_shapes.spatialdata.zarr
            zarr_path = Path(__file__).parent / "output" / "roi_shapes.spatialdata.zarr"
        
        if not zarr_path.exists():
            logger.error(f"SpatialData Zarr file not found: {zarr_path}")
            return jsonify({"error": "SpatialData Zarr file not found"}), 404
        
        # Check if the specific path exists within the Zarr file
        import zarr
        try:
            zarr_group = zarr.open(str(zarr_path), mode='r')
            logger.info(f"Zarr group opened successfully: {zarr_group}")
            
            # Return a mock response for now - in production you'd serve actual Zarr data
            return jsonify({
                "status": "success",
                "zarr_path": str(zarr_path),
                "requested_path": path_param,
                "message": "SpatialData Zarr file found and accessible",
                "zarr_keys": list(zarr_group.keys()) if hasattr(zarr_group, 'keys') else []
            })
            
        except Exception as zarr_error:
            logger.error(f"Error opening Zarr file: {zarr_error}")
            return jsonify({
                "status": "error",
                "message": f"Error opening Zarr file: {zarr_error}",
                "zarr_path": str(zarr_path)
            }), 500
        
    except Exception as e:
        logger.error(f"Error serving SpatialData Zarr: {e}", exc_info=True)
        return jsonify({"error": f"Failed to serve SpatialData Zarr: {e}"}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000) 