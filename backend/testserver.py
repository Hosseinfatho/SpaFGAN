# Backend server code
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
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Constants for Zarr Access
ZARR_BASE_URL = "s3://lsp-public-data/yapp-2023-3d-melanoma/Dataset1-LSP13626-melanoma-in-situ"
ZARR_IMAGE_GROUP_PATH = "0"
TARGET_RESOLUTION_PATH = "3"

# Local data paths
LOCAL_ZARR_PATH = os.path.abspath("input/selected_channels.zarr")
LOCAL_ROI_ZARR_PATH = os.path.abspath("output/roi_shapes.spatialdata.zarr")

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


# #######do not deletet this commented function
# def generate_bio_med_vis_config():
#     """Generate a valid Vitessce config for BioMedVis Challenge with image channels and ROI segmentations using zarr file."""

#     dataset_uid = "A"
#     image_layer_id = "init_A_image_0"
#     segmentation_layer_id = "init_A_obsSegmentations_0"

#     # Image channels configuration (preserved as requested)
#     image_channels = {
#         "init_A_image_0": {"window": [300, 20000], "targetC": 19, "color": [0, 255, 0]},      # CD31
#         "init_A_image_1": {"window": [1000, 7000], "targetC": 27, "color": [255, 255, 0]},     # CD20
#         "init_A_image_2": {"window": [700, 6000], "targetC": 37, "color": [255, 0, 255]},      # CD11b
#         "init_A_image_3": {"window": [1638, 10000], "targetC": 25, "color": [0, 255, 255]},    # CD4
#         "init_A_image_4": {"window": [370, 1432], "targetC": 42, "color": [128, 0, 128]},      # Catalase
#         "init_A_image_5": {"window": [3000, 10000], "targetC": 59, "color": [128, 128, 0]}     # CD11c
#     }

#     # ROI interaction types and colors
#     roi_interactions = [
#         "B-cell infiltration",
#         "Inflammatory zone", 
#         "T-cell entry site",
#         "Oxidative stress niche"
#     ]
    
#     roi_colors = [
#         [0, 200, 200],    # B-cell infiltration - Cyan
#         [0, 255, 0],      # Inflammatory zone - Green
#         [0, 0, 255],      # T-cell entry site - Blue
#         [255, 255, 0]     # Oxidative stress niche - Yellow
#     ]

#     # Create segmentation channel IDs
#     segmentation_channels = [
#         "init_A_obsSegmentations_0",
#         "init_A_obsSegmentations_1", 
#         "init_A_obsSegmentations_2",
#         "init_A_obsSegmentations_3"
#     ]

#     # Build coordination space
#     coordination_space = {
#         "dataset": {dataset_uid: dataset_uid},
#         "imageLayer": {image_layer_id: "__dummy__"},
#         "photometricInterpretation": {image_layer_id: "BlackIsZero"},
#         "spatialRenderingMode": {image_layer_id: "3D"},
#         "spatialLayerOpacity": {image_layer_id: 1.0},
#         "spatialLayerVisible": {image_layer_id: True},
#         "spatialTargetResolution": {image_layer_id: 3},
#         "spatialTargetT": {image_layer_id: 0},
#         "spatialTargetX": {image_layer_id: 5454},
#         "spatialTargetY": {image_layer_id: 2754},
#         "spatialTargetZ": {image_layer_id: 0},
#         "spatialZoom": {image_layer_id: -3.0},
#         "spotLayer": {"init_A_obsSpots_0": "__dummy__"},
#         "segmentationLayer": {segmentation_layer_id: "__dummy__"},
#         "imageChannel": {},
#         "spatialChannelColor": {},
#         "spatialChannelOpacity": {},
#         "spatialChannelVisible": {},
#         "spatialChannelWindow": {},
#         "spatialTargetC": {},
#         "segmentationChannel": {},
#         "obsType": {
#             "init_A_obsSegmentations_0": "B-cell infiltration",
#             "init_A_obsSegmentations_1": "Inflammatory zone", 
#             "init_A_obsSegmentations_2": "T-cell entry site",
#             "init_A_obsSegmentations_3": "Oxidative stress niche",
#             "A": "spot"
#         },
#         "obsColorEncoding": {
#             "init_A_obsSegmentations_0": "spatialChannelColor",
#             "init_A_obsSegmentations_1": "spatialChannelColor",
#             "init_A_obsSegmentations_2": "spatialChannelColor",
#             "init_A_obsSegmentations_3": "spatialChannelColor",
#             "A": "geneSelection"
#         },
#         "spatialSegmentationLayer": {
#             "A": {
#                 "radius": 200,
#                 "stroked": True,
#                 "visible": True,
#                 "opacity": 0.8,
#                 "color": [0, 200, 200]  # Cyan for B-cell infiltration
#             },
#             "B": {
#                 "radius": 200,
#                 "stroked": True,
#                 "visible": True,
#                 "opacity": 0.8,
#                 "color": [0, 255, 0]  # Green for Inflammatory zone
#             },
#             "C": {
#                 "radius": 200,
#                 "stroked": True,
#                 "visible": True,
#                 "opacity": 0.8,
#                 "color": [0, 0, 255]  # Blue for T-cell entry site
#             },
#             "D": {
#                 "radius": 200,
#                 "stroked": True,
#                 "visible": True,
#                 "opacity": 0.8,
#                 "color": [255, 255, 0]  # Yellow for Oxidative stress niche
#             }
#         },
#         "metaCoordinationScopes": {
#             image_layer_id: {
#                 "imageLayer": image_layer_id,
#                 "spatialRenderingMode": image_layer_id,
#                 "spatialLayerOpacity": image_layer_id,
#                 "spatialLayerVisible": image_layer_id,
#                 "spatialTargetResolution": image_layer_id,
#                 "spatialTargetT": image_layer_id,
#                 "spatialTargetX": image_layer_id,
#                 "spatialTargetY": image_layer_id,
#                 "spatialTargetZ": image_layer_id,
#                 "spatialZoom": image_layer_id
#             },
#             "init_A_obsSpots_0": {
#                 "spotLayer": "init_A_obsSpots_0"
#             },
#             segmentation_layer_id: {
#                 "segmentationLayer": [segmentation_layer_id]
#             }
#         },
#         "metaCoordinationScopesBy": {
#             image_layer_id: {
#                 "imageLayer": {
#                     "imageChannel": {image_layer_id: list(image_channels.keys())},
#                     "photometricInterpretation": {image_layer_id: image_layer_id},
#                     "spatialLayerOpacity": {image_layer_id: image_layer_id},
#                     "spatialLayerVisible": {image_layer_id: image_layer_id},
#                     "spatialTargetResolution": {image_layer_id: image_layer_id},
#                     "spatialRenderingMode": {image_layer_id: image_layer_id},
#                     "spatialTargetT": {image_layer_id: image_layer_id},
#                     "spatialTargetX": {image_layer_id: image_layer_id},
#                     "spatialTargetY": {image_layer_id: image_layer_id},
#                     "spatialTargetZ": {image_layer_id: image_layer_id},
#                     "spatialZoom": {image_layer_id: image_layer_id}
#                 },
#                 "imageChannel": {
#                     "spatialChannelColor": {},
#                     "spatialChannelOpacity": {},
#                     "spatialChannelVisible": {},
#                     "spatialChannelWindow": {},
#                     "spatialTargetC": {}
#                 }
#             },
#             "init_A_obsSpots_0": {
#                 "spotLayer": {
#                     "obsColorEncoding": {"init_A_obsSpots_0": "A"},
#                     "featureSelection": {"init_A_obsSpots_0": "A"}
#                 }
#             },
#             segmentation_layer_id: {
#                 "segmentationLayer": {
#                     "segmentationChannel": {
#                         segmentation_layer_id: segmentation_channels
#                     }
#                 },
#                 "segmentationChannel": {
#                     "obsType": {},
#                     "obsColorEncoding": {},
#                     "spatialTargetC": {},
#                     "spatialChannelVisible": {},
#                     "spatialChannelOpacity": {},
#                     "spatialChannelColor": {}
#                 }
#             }
#         }
#     }

#     # Add image channels (preserved as requested)
#     for ch_id, props in image_channels.items():
#         coordination_space["imageChannel"][ch_id] = "__dummy__"
#         coordination_space["spatialChannelColor"][ch_id] = props["color"]
#         coordination_space["spatialChannelOpacity"][ch_id] = 1.0
#         coordination_space["spatialChannelVisible"][ch_id] = True
#         coordination_space["spatialChannelWindow"][ch_id] = props["window"]
#         coordination_space["spatialTargetC"][ch_id] = props["targetC"]

#         for key in ["spatialChannelColor", "spatialChannelOpacity", "spatialChannelVisible", "spatialChannelWindow", "spatialTargetC"]:
#             coordination_space["metaCoordinationScopesBy"][image_layer_id]["imageChannel"][key][ch_id] = ch_id

#     # Add segmentation channels for ROI interactions
#     for idx, (interaction, color) in enumerate(zip(roi_interactions, roi_colors)):
#         channel_id = segmentation_channels[idx]
        
#         coordination_space["segmentationChannel"][channel_id] = "__dummy__"
#         coordination_space["spatialTargetC"][channel_id] = idx
#         coordination_space["spatialChannelColor"][channel_id] = color
#         coordination_space["spatialChannelOpacity"][channel_id] = 0.7
#         coordination_space["spatialChannelVisible"][channel_id] = True

#         # Add to meta coordination scopes
#         coordination_space["metaCoordinationScopesBy"][segmentation_layer_id]["segmentationChannel"]["obsType"][channel_id] = channel_id
#         coordination_space["metaCoordinationScopesBy"][segmentation_layer_id]["segmentationChannel"]["obsColorEncoding"][channel_id] = channel_id
#         coordination_space["metaCoordinationScopesBy"][segmentation_layer_id]["segmentationChannel"]["spatialTargetC"][channel_id] = channel_id
#         coordination_space["metaCoordinationScopesBy"][segmentation_layer_id]["segmentationChannel"]["spatialChannelVisible"][channel_id] = dataset_uid
#         coordination_space["metaCoordinationScopesBy"][segmentation_layer_id]["segmentationChannel"]["spatialChannelOpacity"][channel_id] = dataset_uid
#         coordination_space["metaCoordinationScopesBy"][segmentation_layer_id]["segmentationChannel"]["spatialChannelColor"][channel_id] = channel_id

#     # Dataset files configuration
#     dataset_files = [
#         {
#             "fileType": "image.ome-zarr",
#             "url": "http://localhost:5000/api/spatialdata_zarr/selected_channels.zarr"
#         },
#         {
#             "fileType": "obsSpots.spatialdata.zarr",
#             "url": "http://localhost:5000/api/spatialdata_zarr/roi_shapes.spatialdata.zarr",
#             "coordinationValues": {
#                 "obsType": "spot"
#             },
#             "options": {
#                 "path": "shapes/roi_circles",
#                 "tablePath": "table/table"
#             }
#         }
#     ]

#     return {
#         "version": "1.0.16",
#         "name": "BioMedVis Challenge - Image and ROI Segmentation",
#         "description": "Render base image with marker channels and ROI segmentation layer using zarr file",
#         "initStrategy": "auto",
#         "datasets": [{
#             "uid": dataset_uid,
#             "name": "Blood Vessel",
#             "files": dataset_files
#         }],
#         "coordinationSpace": coordination_space,
#         "layout": [
#             {
#                 "component": "spatialBeta",
#                 "coordinationScopes": {
#                     "dataset": dataset_uid,
#                     "imageLayer": image_layer_id,
#                     "segmentationLayer": segmentation_layer_id,
#                     "metaCoordinationScopes": ["init_A_obsSpots_0", image_layer_id, segmentation_layer_id],
#                     "metaCoordinationScopesBy": ["init_A_obsSpots_0", image_layer_id, segmentation_layer_id],
#                     "spatialSegmentationLayer": ["A", "B", "C", "D"],
#                     "obsType": ["A", "B", "C", "D"],
#                     "obsColorEncoding": ["A", "B", "C", "D"],
#                     "imageChannel": list(image_channels.keys()),
#                     "spatialChannelColor": list(image_channels.keys()),
#                     "spatialChannelOpacity": list(image_channels.keys()),
#                     "spatialChannelVisible": list(image_channels.keys()),
#                     "spatialChannelWindow": list(image_channels.keys()),
#                     "spatialTargetC": list(image_channels.keys()),
#                     "spatialRenderingMode": image_layer_id,
#                     "spatialLayerOpacity": image_layer_id,
#                     "spatialLayerVisible": image_layer_id,
#                     "spatialTargetResolution": image_layer_id,
#                     "spatialTargetT": image_layer_id,
#                     "spatialTargetX": image_layer_id,
#                     "spatialTargetY": image_layer_id,
#                     "spatialTargetZ": image_layer_id,
#                     "spatialZoom": image_layer_id
#                 },
#                 "x": 0, "y": 0, "w": 8, "h": 12
#             },
#             {
#                 "component": "layerControllerBeta",
#                 "coordinationScopes": {
#                     "dataset": dataset_uid,
#                     "imageLayer": image_layer_id,
#                     "segmentationLayer": segmentation_layer_id,
#                     "metaCoordinationScopes": ["init_A_obsSpots_0", image_layer_id, segmentation_layer_id],
#                     "metaCoordinationScopesBy": ["init_A_obsSpots_0", image_layer_id, segmentation_layer_id],
#                     "spatialSegmentationLayer": ["A", "B", "C", "D"],
#                     "obsType": ["A", "B", "C", "D"],
#                     "imageChannel": list(image_channels.keys()),
#                     "spatialChannelColor": list(image_channels.keys()),
#                     "spatialChannelOpacity": list(image_channels.keys()),
#                     "spatialChannelVisible": list(image_channels.keys()),
#                     "spatialChannelWindow": list(image_channels.keys()),
#                     "spatialTargetC": list(image_channels.keys()),
#                     "spatialRenderingMode": image_layer_id,
#                     "spatialLayerOpacity": image_layer_id,
#                     "spatialLayerVisible": image_layer_id,
#                     "spatialTargetResolution": image_layer_id,
#                     "spatialTargetT": image_layer_id,
#                     "spatialTargetX": image_layer_id,
#                     "spatialTargetY": image_layer_id,
#                     "spatialTargetZ": image_layer_id,
#                     "spatialZoom": image_layer_id
#                 },
#                 "x": 8, "y": 0, "w": 4, "h": 12
#             }
#         ]
#     }
# ## end config generator


def generate_vitessce_config():
    """Generate Vitessce configuration exactly like simplere.json"""
    
    # Create the config dictionary exactly like simplere.json
    config = {
        "version": "1.0.16",
        "name": "SpaFGAN Interactions",
        "description": "4teractions: B-cell infiltration, Inflammatory zone, Oxidative stress niche, T-cell entry site",
        "datasets": [
            {
                "uid": "spafgan",
                "name": "SpaFGAN Dataset",
                "files": [
                    {
                        "fileType": "obsSegmentations.ome-tiff",
                        "url": "http://localhost:5000/api/segmentation/obsSegmentations.ome.tiff",
                        "options": {
                            "offsetsUrl": "http://localhost:5000/api/segmentation/obsSegmentations.offsets.json"
                        },
                        # "coordinationValues": {
                        #     "fileUid": "spafgan"
                        # }
                    }
                ]
            }
        ],
        "initStrategy": "auto",
        "coordinationSpace": {
            "fileUid": {
                "bitmask": "spafgan"
            },
            "segmentationLayer": {
                "ml": "ml"
            },
            "segmentationChannel": {
                "bci": "bci",
                "iz": "iz",
                "osn": "osn",
                "tces": "tces"
            },
            "obsType": {
                "bci": "B-cell infiltration",
                "iz": "Inflammatory zone",
                "osn": "Oxidative stress niche",
                "tces": "T-cell entry site"
            },
            "obsColorEncoding": {
                "bci": "spatialChannelColor",
                "iz": "spatialChannelColor",
                "osn": "spatialChannelColor",
                "tces": "spatialChannelColor"
            },
            "featureValueColormap": {
                "bci": "plasma",
                "iz": "plasma",
                "osn": "plasma",
                "tces": "plasma"
            },
            "featureValueColormapRange": {
                "bci": [0, 1],
                "iz": [0, 1],
                "osn": [0, 1],
                "tces": [0, 1]
            },
            "featureType": {
                "global": "feature"
            },
            "featureValueType": {
                "global": "value"
            },
            "featureSelection": {
                "bci": None,
                "iz": None,
                "osn": None,
                "tces": None
            },
            "spatialTargetC": {
                "bci": 0,
                "iz": 1,
                "osn": 2,
                "tces": 3
            },
            "spatialChannelColor": {
                "bci": [0, 200, 200],      # Cyan
                "iz": [0, 255, 0],         # Green
                "osn": [255, 255, 0],      # Yellow
                "tces": [0, 0, 255]        # Blue
            },
            "spatialChannelVisible": {
                "bci": True,
                "iz": True,
                "osn": True,
                "tces": True
            },
            "spatialLayerVisible": {
                "bitmask": True
            },
            "spatialChannelOpacity": {
                "bci": 1,
                "iz": 1,
                "osn": 1,
                "tces": 1
            },
            "spatialLayerOpacity": {
                "bitmask": 1
            },
            "spatialSegmentationFilled": {
                "bci": True,
                "iz": True,
                "osn": True,
                "tces": True
            },
            "spatialSegmentationStrokeWidth": {
                "bci": 1,
                "iz": 1,
                "osn": 1,
                "tces": 1
            },
            "obsHighlight": {
                "bci": None,
                "iz": None,
                "osn": None,
                "tces": None
            },
            "spatialTargetX": {
                "A": 5500
            },
            "spatialTargetY": {
                "A": 2700
            },
            "spatialZoom": {
                "A": -3.0
            },
            "metaCoordinationScopes": {
            "metaA": {
                    "obsType": ["bci", "iz", "osn", "tces"],
                    "segmentationLayer": ["ml"]
            }
        },
            "metaCoordinationScopesBy": {
            "metaA": {
                "segmentationLayer": {
                        "fileUid": {
                            "ml": "bitmask"
                },
                "segmentationChannel": {
                            "ml": ["bci", "iz", "osn", "tces"]
                        },
                        "spatialLayerVisible": {
                            "ml": "bitmask"
                        },
                        "spatialLayerOpacity": {
                            "ml": "bitmask"
                        }
                    },
                    "segmentationChannel": {
                        "obsType": {
                            "bci": "bci",
                            "iz": "iz",
                            "osn": "osn",
                            "tces": "tces"
                        },
                        "featureType": {
                            "bci": "global",
                            "iz": "global",
                            "osn": "global",
                            "tces": "global"
                        },
                        "featureValueType": {
                            "bci": "global",
                            "iz": "global",
                            "osn": "global",
                            "tces": "global"
                        },
                        "featureSelection": {
                            "bci": "bci",
                            "iz": "iz",
                            "osn": "osn",
                            "tces": "tces"
                        },
                        "spatialTargetC": {
                            "bci": "bci",
                            "iz": "iz",
                            "osn": "osn",
                            "tces": "tces"
                        },
                        "obsColorEncoding": {
                            "bci": "bci",
                            "iz": "iz",
                            "osn": "osn",
                            "tces": "tces"
                        },
                        "featureValueColormap": {
                            "bci": "bci",
                            "iz": "iz",
                            "osn": "osn",
                            "tces": "tces"
                        },
                        "featureValueColormapRange": {
                            "bci": "bci",
                            "iz": "iz",
                            "osn": "osn",
                            "tces": "tces"
                        },
                        "spatialChannelVisible": {
                            "bci": "bci",
                            "iz": "iz",
                            "osn": "osn",
                            "tces": "tces"
                        },
                        "spatialChannelOpacity": {
                            "bci": "bci",
                            "iz": "iz",
                            "osn": "osn",
                            "tces": "tces"
                        },
                        "spatialChannelColor": {
                            "bci": "bci",
                            "iz": "iz",
                            "osn": "osn",
                            "tces": "tces"
                        },
                        "spatialSegmentationFilled": {
                            "bci": "bci",
                            "iz": "iz",
                            "osn": "osn",
                            "tces": "tces"
                        },
                        "spatialSegmentationStrokeWidth": {
                            "bci": "bci",
                            "iz": "iz",
                            "osn": "osn",
                            "tces": "tces"
                        },
                        "obsHighlight": {
                            "bci": "bci",
                            "iz": "iz",
                            "osn": "osn",
                            "tces": "tces"
                        }
                    }
                }
            }
        },
        "layout": [
            {
                "component": "spatialBeta",
                "coordinationScopes": {
                    "metaCoordinationScopes": ["metaA"],
                    "metaCoordinationScopesBy": ["metaA"],
                    "spatialTargetX": "A",
                    "spatialTargetY": "A",
                    "spatialZoom": "A"
                },
                "x": 0,
                "y": 0,
                "w": 8,
                "h": 12
            },
            {
                "component": "layerControllerBeta",
                "coordinationScopes": {
                    "metaCoordinationScopes": ["metaA"],
                    "metaCoordinationScopesBy": ["metaA"],
                    "spatialTargetX": "A",
                    "spatialTargetY": "A",
                    "spatialZoom": "A"
                },
                "x": 8,
                "y": 0,
                "w": 4,
                "h": 12
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

from flask import send_from_directory, jsonify
from pathlib import Path

# Define absolute path to the .sdata.zarr directory
ZARR_BASE_DIR = Path("D:/VIS2025/BIoVisChallenges/SpaFGAN/backend/output/roi_shapes.spatialdata.zarr")

@app.route("/api/zarr/<path:filename>", methods=["GET"])
def serve_zarr_file(filename):
    """Serve internal files from the Zarr store directory"""
    try:
        return send_from_directory(ZARR_BASE_DIR, filename)
    except Exception as e:
        logger.error(f"Error serving Zarr file '{filename}': {e}", exc_info=True)
        return jsonify({"error": f"Failed to serve Zarr file '{filename}': {e}"}), 500

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
        
        # Keys are already correct, return as is
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
        
        # Keys are already correct, return as is
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
        
        # Keys are already correct, return as is
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
        
        # Keys are already correct, return as is
        return jsonify(roi_data)
        
    except Exception as e:
        logger.error(f"Error serving roi_segmentation_Oxidative_stress_niche.json: {e}", exc_info=True)
        return jsonify({"error": f"Failed to serve roi_segmentation_Oxidative_stress_niche.json: {e}"}), 500

@app.route('/api/roi_segmentation_combined.json', methods=['GET'])
def get_roi_segmentation_combined():
    """Serve combined ROI segmentation data with all interactions"""
    logger.info("Request received for /api/roi_segmentation_combined.json [GET]")
    
    try:
        # Load all ROI files with proper mapping
        roi_files = {
            'B-cell_infiltration': 'roi_segmentation_B-cell_infiltration.json',
            'Inflammatory_zone': 'roi_segmentation_Inflammatory_zone.json',
            'T-cell_entry_site': 'roi_segmentation_T-cell_entry_site.json',
            'Oxidative_stress_niche': 'roi_segmentation_Oxidative_stress_niche.json'
        }
        
        combined_data = {}
        
        for interaction_name, filename in roi_files.items():
            file_path = Path(__file__).parent / 'output' / filename
            
            if file_path.exists():
                with open(file_path, 'r') as f:
                    roi_data = json.load(f)
                    
                    # Map the ROI keys to the interaction name
                    for roi_key, roi_coordinates in roi_data.items():
                        # Create a new key that matches the obsType
                        new_key = roi_key.replace('ROI_Inflammatory_B-cell_', f'{interaction_name}_')
                        new_key = new_key.replace('ROI_T-Cell_', f'{interaction_name}_')
                        new_key = new_key.replace('ROI_Inflammatory_', f'{interaction_name}_')
                        new_key = new_key.replace('ROI_Oxidative_', f'{interaction_name}_')
                        
                        combined_data[new_key] = roi_coordinates
                    
                    logger.info(f"Loaded {len(roi_data)} ROIs from {filename}")
            else:
                logger.warning(f"File not found: {filename}")
        
        return jsonify(combined_data)
        
    except Exception as e:
        logger.error(f"Error serving combined ROI segmentation: {e}", exc_info=True)
        return jsonify({"error": f"Failed to serve combined ROI segmentation: {e}"}), 500

@app.route('/api/roi_segmentation_filtered.json', methods=['GET'])
def get_roi_segmentation_filtered():
    """Serve filtered ROI segmentation data based on query parameter"""
    logger.info("Request received for /api/roi_segmentation_filtered.json [GET]")
    
    try:
        # Get the interaction type from query parameter
        interaction_type = request.args.get('type', 'B-cell_infiltration')
        
        # Map interaction type to filename
        roi_files = {
            'B-cell_infiltration': 'roi_segmentation_B-cell_infiltration.json',
            'Inflammatory_zone': 'roi_segmentation_Inflammatory_zone.json',
            'T-cell_entry_site': 'roi_segmentation_T-cell_entry_site.json',
            'Oxidative_stress_niche': 'roi_segmentation_Oxidative_stress_niche.json'
        }
        
        if interaction_type not in roi_files:
            return jsonify({"error": f"Unknown interaction type: {interaction_type}"}), 400
        
        filename = roi_files[interaction_type]
        file_path = Path(__file__).parent / 'output' / filename
        
        if not file_path.exists():
            return jsonify({"error": f"File not found: {filename}"}), 404
        
        with open(file_path, 'r') as f:
            roi_data = json.load(f)
        
        # Map the ROI keys to the interaction name
        filtered_data = {}
        for roi_key, roi_coordinates in roi_data.items():
            new_key = roi_key.replace('ROI_Inflammatory_B-cell_', f'{interaction_type}_')
            new_key = new_key.replace('ROI_T-Cell_', f'{interaction_type}_')
            new_key = new_key.replace('ROI_Inflammatory_', f'{interaction_type}_')
            new_key = new_key.replace('ROI_Oxidative_', f'{interaction_type}_')
            
            filtered_data[new_key] = roi_coordinates
        
        logger.info(f"Served {len(filtered_data)} ROIs for {interaction_type}")
        return jsonify(filtered_data)
        
    except Exception as e:
        logger.error(f"Error serving filtered ROI segmentation: {e}", exc_info=True)
        return jsonify({"error": f"Failed to serve filtered ROI segmentation: {e}"}), 500

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

@app.route("/api/local_image", methods=["GET"])
def serve_local_image():
    """Serve local image data from selected_channels.zarr"""
    try:
        if os.path.exists(LOCAL_ZARR_PATH):
            return send_from_directory("input", "selected_channels.zarr")
        else:
            return jsonify({"error": "Local image data not found"}), 404
    except Exception as e:
        return jsonify({"error": f"Failed to serve local image: {e}"}), 500

@app.route("/api/local_roi", methods=["GET"])
def serve_local_roi():
    """Serve local ROI data from roi_shapes.spatialdata.zarr"""
    try:
        if os.path.exists(LOCAL_ROI_ZARR_PATH):
            return send_from_directory("output", "roi_shapes.spatialdata.zarr")
        else:
            return jsonify({"error": "Local ROI data not found"}), 404
    except Exception as e:
        return jsonify({"error": f"Failed to serve local ROI: {e}"}), 500

@app.route("/api/spatialdata_zarr/<path:filename>", methods=["GET"])
def serve_spatialdata_zarr(filename):
    """Serve SpatialData Zarr files and their internal files"""
    try:
        logger.info(f"Serving SpatialData Zarr file: {filename}")
        
        # Handle nested paths within Zarr files
        if "/" in filename:
            # This is a request for internal Zarr files (e.g., "selected_channels.zarr/.zattrs")
            zarr_name, internal_path = filename.split("/", 1)
            
            if zarr_name == "selected_channels.zarr":
                input_dir = Path(__file__).parent / "input" / "selected_channels.zarr"
                if (input_dir / internal_path).exists():
                    return send_from_directory(input_dir, internal_path)
                else:
                    logger.error(f"Internal file not found: {input_dir / internal_path}")
                    return jsonify({"error": f"Internal file '{internal_path}' not found"}), 404
            elif zarr_name == "roi_shapes.spatialdata.zarr":
                output_dir = Path(__file__).parent / "output" / "roi_shapes.spatialdata.zarr"
                if (output_dir / internal_path).exists():
                    return send_from_directory(output_dir, internal_path)
                else:
                    logger.error(f"Internal file not found: {output_dir / internal_path}")
                    return jsonify({"error": f"Internal file '{internal_path}' not found"}), 404
            else:
                return jsonify({"error": f"Zarr file '{zarr_name}' not supported"}), 404
        else:
            # This is a request for the main Zarr directory
            if filename == "selected_channels.zarr":
                input_dir = Path(__file__).parent / "input" / "selected_channels.zarr"
                if input_dir.exists():
                    # Return a simple JSON response indicating this is a Zarr directory
                    return jsonify({
                        "type": "zarr_directory",
                        "name": filename,
                        "path": str(input_dir)
                    })
                else:
                    logger.error(f"Directory not found: {input_dir}")
                    return jsonify({"error": f"Zarr directory '{filename}' not found"}), 404
            elif filename == "roi_shapes.spatialdata.zarr":
                output_dir = Path(__file__).parent / "output" / "roi_shapes.spatialdata.zarr"
                if output_dir.exists():
                    # Return a simple JSON response indicating this is a Zarr directory
                    return jsonify({
                        "type": "zarr_directory",
                        "name": filename,
                        "path": str(output_dir)
                    })
                else:
                    logger.error(f"Directory not found: {output_dir}")
                    return jsonify({"error": f"Zarr directory '{filename}' not found"}), 404
            else:
                return jsonify({"error": f"Zarr file '{filename}' not supported"}), 404
    except Exception as e:
        logger.error(f"Error serving SpatialData Zarr file '{filename}': {e}", exc_info=True)
        return jsonify({"error": f"Failed to serve Zarr file '{filename}': {e}"}), 500

@app.route("/api/annodata_zarr/<path:filename>", methods=["GET"])
def serve_annodata_zarr(filename):
    """Serve AnnData-Zarr files from vitessce_files directory"""
    try:
        logger.info(f"Serving AnnData Zarr file: {filename}")
        
        # Handle nested paths within Zarr files
        if "/" in filename:
            # This is a request for internal Zarr files (e.g., "B_cell_infiltration_obsFeatureMatrix.anndata.zarr/.zattrs")
            zarr_name, internal_path = filename.split("/", 1)
            
            zarr_dir = Path(__file__).parent / "output" / "vitessce_files" / zarr_name
            if (zarr_dir / internal_path).exists():
                return send_from_directory(zarr_dir, internal_path)
            else:
                logger.error(f"Internal file not found: {zarr_dir / internal_path}")
                return jsonify({"error": f"Internal file '{internal_path}' not found"}), 404
        else:
            # This is a request for the main Zarr directory
            zarr_dir = Path(__file__).parent / "output" / "vitessce_files" / filename
            if zarr_dir.exists():
                # Return a simple JSON response indicating this is a Zarr directory
                return jsonify({
                    "type": "zarr_directory",
                    "name": filename,
                    "path": str(zarr_dir)
                })
            else:
                logger.error(f"Directory not found: {zarr_dir}")
                return jsonify({"error": f"Zarr directory '{filename}' not found"}), 404
    except Exception as e:
        logger.error(f"Error serving AnnData Zarr file '{filename}': {e}", exc_info=True)
        return jsonify({"error": f"Failed to serve Zarr file '{filename}': {e}"}), 500

@app.route("/api/segmentation/<path:filename>", methods=["GET"])
def serve_segmentation_file(filename):
    """Serve segmentation files (ome-tif and offsets)"""
    try:
        logger.info(f"Serving segmentation file: {filename}")
        
        # First try vitessce_files directory
        segmentation_dir = Path(__file__).parent / "output" / "vitessce_files"
        file_path = segmentation_dir / filename
        
        if file_path.exists():
            # Disable range requests to avoid 416 errors
            response = send_from_directory(segmentation_dir, filename)
            response.headers['Accept-Ranges'] = 'none'
            return response
        else:
            # If not found in vitessce_files, try output directory
            segmentation_dir = Path(__file__).parent / "output"
            file_path = segmentation_dir / filename
            
            if file_path.exists():
                # Disable range requests to avoid 416 errors
                response = send_from_directory(segmentation_dir, filename)
                response.headers['Accept-Ranges'] = 'none'
                return response
            else:
                logger.error(f"Segmentation file not found: {file_path}")
                return jsonify({"error": f"Segmentation file '{filename}' not found"}), 404
    except Exception as e:
        logger.error(f"Error serving segmentation file '{filename}': {e}", exc_info=True)
        return jsonify({"error": f"Failed to serve segmentation file '{filename}': {e}"}), 500



if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000) 