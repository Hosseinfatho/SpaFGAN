# Backend server code
import json
import logging
from flask_cors import CORS
from flask import Flask, jsonify, send_from_directory, request
import os

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

# ## roi with json file but only show last file
# def generate_vitessce_config1():
#     """Generate Vitessce configuration in Python based on sampleconfig.json structure"""
    
#     # Image channels configuration (like Image0 in sampleconfig)
#     image_channels = {
#         'CD31': {'id': 'cd31', 'color': [0, 255, 0], 'window': [300, 20000], 'targetC': 19},      # Green
#         'CD20': {'id': 'cd20', 'color': [255, 255, 0], 'window': [1000, 7000], 'targetC': 27},    # Yellow
#         'CD11b': {'id': 'cd11b', 'color': [255, 0, 255], 'window': [700, 6000], 'targetC': 37},  # Magenta
#         'CD4': {'id': 'cd4', 'color': [0, 255, 255], 'window': [1638, 10000], 'targetC': 25},    # Cyan
#         'CD11c': {'id': 'cd11c', 'color': [128, 0, 128], 'window': [370, 1432], 'targetC': 42}   # Purple
#     }

#     # ROI interaction types (your actual interaction names) - using OrderedDict to maintain order
#     roi_interactions = OrderedDict([
#         ('B-cell_infiltration', {'id': 'bcell', 'color': [211, 94, 26]}),
#         ('Inflammatory_zone', {'id': 'inflammatory', 'color': [228, 158, 37]}),
#         ('T-cell_entry_site', {'id': 'tcell', 'color': [239, 226, 82]}),
#         ('Oxidative_stress_niche', {'id': 'oxidative', 'color': [22, 157, 116]})
#     ])

#     # Build coordination space following sampleconfig.json structure
#     coordination_space = {
#         'dataset': {"A": "bv"},
#         'imageLayer': {"histology": "histology"},
#         'segmentationLayer': {"ml": "ml"},
#         'imageChannel': {},
#         'segmentationChannel': {},
#         'obsType': {},
#         'obsColorEncoding': {},
#         'spatialChannelColor': {},
#         'spatialChannelOpacity': {},
#         'spatialChannelVisible': {},
#         'spatialChannelWindow': {},
#         'spatialTargetC': {},
#         'spatialLayerOpacity': {"histology": 1.0, "ml": 1.0},
#         'spatialLayerVisible': {"histology": True, "ml": True},
#         'spatialTargetX': {"A": 5454},
#         'spatialTargetY': {"A": 2754},
#         'spatialZoom': {"A": -3.0},
#         'photometricInterpretation': {"histology": "BlackIsZero"},
#         'spatialSegmentationFilled': {},
#         'spatialSegmentationStrokeWidth': {},
#         'metaCoordinationScopes': {
#             "metaA": {
#                 "obsType": ["B-cell_infiltration", "Inflammatory_zone", "T-cell_entry_site", "Oxidative_stress_niche"],
#                 "segmentationLayer": ["ml"],
#                 "imageLayer": ["histology"],
#                 "spatialChannelVisible": ["B-cell_infiltration", "Inflammatory_zone", "T-cell_entry_site", "Oxidative_stress_niche"],
#                 "spatialChannelOpacity": ["B-cell_infiltration", "Inflammatory_zone", "T-cell_entry_site", "Oxidative_stress_niche"],
#                 "spatialChannelColor": ["B-cell_infiltration", "Inflammatory_zone", "T-cell_entry_site", "Oxidative_stress_niche"]
#             }
#         },
#         'metaCoordinationScopesBy': {
#             "metaA": {
#                 "imageLayer": {
#                     "imageChannel": {"histology": ["CD31", "CD20", "CD11b", "CD4", "CD11c"]},
#                     "spatialLayerVisible": {"histology": "histology"},
#                     "spatialLayerOpacity": {"histology": "histology"},
#                     "photometricInterpretation": {"histology": "histology"}
#                 },
#                 "imageChannel": {
#                     "spatialTargetC": {},
#                     "spatialChannelColor": {},
#                     "spatialChannelVisible": {},
#                     "spatialChannelOpacity": {},
#                     "spatialChannelWindow": {}
#                 },
#                 "segmentationLayer": {
#                     "segmentationChannel": {"ml": ["B-cell_infiltration", "Inflammatory_zone", "T-cell_entry_site", "Oxidative_stress_niche"]},
#                     "spatialLayerVisible": {"ml": "ml"},
#                     "spatialLayerOpacity": {"ml": "ml"}
#                 },
#                 "segmentationChannel": {
#                     "obsType": {},
#                     "obsColorEncoding": {},
#                     "spatialTargetC": {},
#                     "spatialChannelVisible": {},
#                     "spatialChannelOpacity": {},
#                     "spatialChannelColor": {},
#                     "spatialSegmentationFilled": {},
#                     "spatialSegmentationStrokeWidth": {}
#                 }
#             }
#         }
#     }

#     # Add image channels (like Image0 in sampleconfig)
#     for ch_name, ch_props in image_channels.items():
#         coordination_space['imageChannel'][ch_name] = "__dummy__"
#         coordination_space['spatialChannelColor'][ch_name] = ch_props['color']
#         coordination_space['spatialChannelOpacity'][ch_name] = 1.0
#         coordination_space['spatialChannelVisible'][ch_name] = True
#         coordination_space['spatialChannelWindow'][ch_name] = ch_props['window']
#         coordination_space['spatialTargetC'][ch_name] = ch_props['targetC']
        
#         # Add to meta coordination scopes (like in sampleconfig.json)
#         coordination_space['metaCoordinationScopesBy']['metaA']['imageChannel']['spatialTargetC'][ch_name] = ch_name
#         coordination_space['metaCoordinationScopesBy']['metaA']['imageChannel']['spatialChannelColor'][ch_name] = ch_name
#         coordination_space['metaCoordinationScopesBy']['metaA']['imageChannel']['spatialChannelVisible'][ch_name] = ch_name
#         coordination_space['metaCoordinationScopesBy']['metaA']['imageChannel']['spatialChannelOpacity'][ch_name] = ch_name
#         coordination_space['metaCoordinationScopesBy']['metaA']['imageChannel']['spatialChannelWindow'][ch_name] = ch_name

#     # Add ROI interactions (like tissue types in sampleconfig)
#     # Use sequential spatialTargetC values starting after image channels
#     for idx, (roi_name, roi_props) in enumerate(roi_interactions.items()):
#         target_c = len(image_channels) + idx  # Start after image channels
#         coordination_space['segmentationChannel'][roi_name] = "__dummy__"
#         coordination_space['obsType'][roi_name] = roi_name
#         coordination_space['obsColorEncoding'][roi_name] = "spatialChannelColor"
#         coordination_space['spatialChannelColor'][roi_name] = roi_props['color']
#         coordination_space['spatialChannelOpacity'][roi_name] = 0.8
#         # Start with B-cell_infiltration ON and others OFF
#         coordination_space['spatialChannelVisible'][roi_name] = (roi_name == 'B-cell_infiltration')
#         coordination_space['spatialTargetC'][roi_name] = target_c
#         coordination_space['spatialSegmentationFilled'][roi_name] = True
#         coordination_space['spatialSegmentationStrokeWidth'][roi_name] = 1
        

        
#         # Add to meta coordination scopes (like in sampleconfig.json)
#         coordination_space['metaCoordinationScopesBy']['metaA']['segmentationChannel']['obsType'][roi_name] = roi_name
#         coordination_space['metaCoordinationScopesBy']['metaA']['segmentationChannel']['obsColorEncoding'][roi_name] = roi_name
#         coordination_space['metaCoordinationScopesBy']['metaA']['segmentationChannel']['spatialTargetC'][roi_name] = roi_name
#         coordination_space['metaCoordinationScopesBy']['metaA']['segmentationChannel']['spatialChannelVisible'][roi_name] = roi_name
#         coordination_space['metaCoordinationScopesBy']['metaA']['segmentationChannel']['spatialChannelOpacity'][roi_name] = roi_name
#         coordination_space['metaCoordinationScopesBy']['metaA']['segmentationChannel']['spatialChannelColor'][roi_name] = roi_name
#         coordination_space['metaCoordinationScopesBy']['metaA']['segmentationChannel']['spatialSegmentationFilled'][roi_name] = roi_name
#         coordination_space['metaCoordinationScopesBy']['metaA']['segmentationChannel']['spatialSegmentationStrokeWidth'][roi_name] = roi_name

#     config = {
#         'version': '1.0.16',
#         'name': 'BioMedVis Challenge - Independent Layers',
#         'description': 'Image channels and ROI interactions with independent control',
#         'datasets': [{
#             'uid': 'bv',
#             'name': 'Blood Vessel',
#             'files': [
#                 {
#                     'fileType': 'image.ome-zarr',
#                     'url': 'https://lsp-public-data.s3.amazonaws.com/yapp-2023-3d-melanoma/Dataset1-LSP13626-melanoma-in-situ/0',
#                     'coordinationValues': {
#                         'imageLayer': 'histology'
#                     }
#                 },
#                 {
#                     'fileType': 'obsSegmentations.json',
#                     'url': 'http://localhost:5000/api/roi_segmentation_B-cell_infiltration.json',
#                     'coordinationValues': {
#                         'obsType': 'B-cell_infiltration',
#                         'segmentationLayer': 'ml'
#                     },
#                 },
#                 {
#                     'fileType': 'obsSegmentations.json',
#                     'url': 'http://localhost:5000/api/roi_segmentation_Inflammatory_zone.json',
#                     'coordinationValues': {
#                         'obsType': 'Inflammatory_zone',
#                         'segmentationLayer': 'ml'
#                     },
#                 },
#                 {
#                     'fileType': 'obsSegmentations.json',
#                     'url': 'http://localhost:5000/api/roi_segmentation_T-cell_entry_site.json',
#                     'coordinationValues': {
#                         'obsType': 'T-cell_entry_site',
#                         'segmentationLayer': 'ml'
#                     },
#                 },
#                 {
#                     'fileType': 'obsSegmentations.json',
#                     'url': 'http://localhost:5000/api/roi_segmentation_Oxidative_stress_niche.json',
#                     'coordinationValues': {
#                         'obsType': 'Oxidative_stress_niche',
#                         'segmentationLayer': 'ml'
#                     },
#                 }
#             ]
#         }],
#         'initStrategy': 'auto',
#         'coordinationSpace': coordination_space,
#         'layout': [
#             {
#                 'component': 'spatialBeta',
#                 'coordinationScopes': {
#                     'metaCoordinationScopes': ["metaA"],
#                     'metaCoordinationScopesBy': ["metaA"],
#                     'spatialTargetX': "A",
#                     'spatialTargetY': "A",
#                     'spatialZoom': "A",
#                     'spatialChannelVisible': ["B-cell_infiltration", "Inflammatory_zone", "T-cell_entry_site", "Oxidative_stress_niche"],
#                     'spatialChannelOpacity': ["B-cell_infiltration", "Inflammatory_zone", "T-cell_entry_site", "Oxidative_stress_niche"],
#                     'spatialChannelColor': ["B-cell_infiltration", "Inflammatory_zone", "T-cell_entry_site", "Oxidative_stress_niche"]
#                 },
#                 'x': 0, 'y': 0, 'w': 8, 'h': 12
#             },
#             {
#                 'component': 'layerControllerBeta',
#                 'coordinationScopes': {
#                     'metaCoordinationScopes': ["metaA"],
#                     'metaCoordinationScopesBy': ["metaA"],
#                     'spatialTargetX': "A",
#                     'spatialTargetY': "A",
#                     'spatialZoom': "A",
#                     'spatialChannelVisible': ["B-cell_infiltration", "Inflammatory_zone", "T-cell_entry_site", "Oxidative_stress_niche"],
#                     'spatialChannelOpacity': ["B-cell_infiltration", "Inflammatory_zone", "T-cell_entry_site", "Oxidative_stress_niche"],
#                     'spatialChannelColor': ["B-cell_infiltration", "Inflammatory_zone", "T-cell_entry_site", "Oxidative_stress_niche"]
#                 },
#                 'x': 0, 'y': 8, 'w': 4, 'h': 12
#             }
#         ]
#     }

#     return config








def generate_bio_med_vis_config():
    """Generate a valid Vitessce config for BioMedVis Challenge with image channels and ROI segmentations using zarr file."""

    dataset_uid = "A"
    image_layer_id = "init_A_image_0"
    segmentation_layer_id = "init_A_obsSegmentations_0"

    # Image channels configuration (preserved as requested)
    image_channels = {
        "init_A_image_0": {"window": [300, 20000], "targetC": 19, "color": [0, 255, 0]},      # CD31
        "init_A_image_1": {"window": [1000, 7000], "targetC": 27, "color": [255, 255, 0]},     # CD20
        "init_A_image_2": {"window": [700, 6000], "targetC": 37, "color": [255, 0, 255]},      # CD11b
        "init_A_image_3": {"window": [1638, 10000], "targetC": 25, "color": [0, 255, 255]},    # CD4
        "init_A_image_4": {"window": [370, 1432], "targetC": 42, "color": [128, 0, 128]},      # Catalase
        "init_A_image_5": {"window": [3000, 10000], "targetC": 59, "color": [128, 128, 0]}     # CD11c
    }

    # ROI interaction types and colors
    roi_interactions = [
        "B-cell infiltration",
        "Inflammatory zone", 
        "T-cell entry site",
        "Oxidative stress niche"
    ]
    
    roi_colors = [
        [0, 200, 200],    # B-cell infiltration - Cyan
        [0, 255, 0],      # Inflammatory zone - Green
        [0, 0, 255],      # T-cell entry site - Blue
        [255, 255, 0]     # Oxidative stress niche - Yellow
    ]

    # Create segmentation channel IDs
    segmentation_channels = [
        "init_A_obsSegmentations_0",
        "init_A_obsSegmentations_1", 
        "init_A_obsSegmentations_2",
        "init_A_obsSegmentations_3"
    ]

    # Build coordination space
    coordination_space = {
        "dataset": {dataset_uid: dataset_uid},
        "imageLayer": {image_layer_id: "__dummy__"},
        "photometricInterpretation": {image_layer_id: "BlackIsZero"},
        "spatialRenderingMode": {image_layer_id: "3D"},
        "spatialLayerOpacity": {image_layer_id: 1.0},
        "spatialLayerVisible": {image_layer_id: True},
        "spatialTargetResolution": {image_layer_id: 3},
        "spatialTargetT": {image_layer_id: 0},
        "spatialTargetX": {image_layer_id: 5454},
        "spatialTargetY": {image_layer_id: 2754},
        "spatialTargetZ": {image_layer_id: 0},
        "spatialZoom": {image_layer_id: -3.0},
        "spotLayer": {"init_A_obsSpots_0": "__dummy__"},
        "segmentationLayer": {segmentation_layer_id: "__dummy__"},
        "imageChannel": {},
        "spatialChannelColor": {},
        "spatialChannelOpacity": {},
        "spatialChannelVisible": {},
        "spatialChannelWindow": {},
        "spatialTargetC": {},
        "segmentationChannel": {},
        "obsType": {
            "init_A_obsSegmentations_0": "B-cell infiltration",
            "init_A_obsSegmentations_1": "Inflammatory zone", 
            "init_A_obsSegmentations_2": "T-cell entry site",
            "init_A_obsSegmentations_3": "Oxidative stress niche",
            "A": "spot"
        },
        "obsColorEncoding": {
            "init_A_obsSegmentations_0": "spatialChannelColor",
            "init_A_obsSegmentations_1": "spatialChannelColor",
            "init_A_obsSegmentations_2": "spatialChannelColor",
            "init_A_obsSegmentations_3": "spatialChannelColor",
            "A": "geneSelection"
        },
        "spatialSegmentationLayer": {
            "A": {
                "radius": 200,
                "stroked": True,
                "visible": True,
                "opacity": 0.8,
                "color": [0, 200, 200]  # Cyan for B-cell infiltration
            },
            "B": {
                "radius": 200,
                "stroked": True,
                "visible": True,
                "opacity": 0.8,
                "color": [0, 255, 0]  # Green for Inflammatory zone
            },
            "C": {
                "radius": 200,
                "stroked": True,
                "visible": True,
                "opacity": 0.8,
                "color": [0, 0, 255]  # Blue for T-cell entry site
            },
            "D": {
                "radius": 200,
                "stroked": True,
                "visible": True,
                "opacity": 0.8,
                "color": [255, 255, 0]  # Yellow for Oxidative stress niche
            }
        },
        "metaCoordinationScopes": {
            image_layer_id: {
                "imageLayer": image_layer_id,
                "spatialRenderingMode": image_layer_id,
                "spatialLayerOpacity": image_layer_id,
                "spatialLayerVisible": image_layer_id,
                "spatialTargetResolution": image_layer_id,
                "spatialTargetT": image_layer_id,
                "spatialTargetX": image_layer_id,
                "spatialTargetY": image_layer_id,
                "spatialTargetZ": image_layer_id,
                "spatialZoom": image_layer_id
            },
            "init_A_obsSpots_0": {
                "spotLayer": "init_A_obsSpots_0"
            },
            segmentation_layer_id: {
                "segmentationLayer": [segmentation_layer_id]
            }
        },
        "metaCoordinationScopesBy": {
            image_layer_id: {
                "imageLayer": {
                    "imageChannel": {image_layer_id: list(image_channels.keys())},
                    "photometricInterpretation": {image_layer_id: image_layer_id},
                    "spatialLayerOpacity": {image_layer_id: image_layer_id},
                    "spatialLayerVisible": {image_layer_id: image_layer_id},
                    "spatialTargetResolution": {image_layer_id: image_layer_id},
                    "spatialRenderingMode": {image_layer_id: image_layer_id},
                    "spatialTargetT": {image_layer_id: image_layer_id},
                    "spatialTargetX": {image_layer_id: image_layer_id},
                    "spatialTargetY": {image_layer_id: image_layer_id},
                    "spatialTargetZ": {image_layer_id: image_layer_id},
                    "spatialZoom": {image_layer_id: image_layer_id}
                },
                "imageChannel": {
                    "spatialChannelColor": {},
                    "spatialChannelOpacity": {},
                    "spatialChannelVisible": {},
                    "spatialChannelWindow": {},
                    "spatialTargetC": {}
                }
            },
            "init_A_obsSpots_0": {
                "spotLayer": {
                    "obsColorEncoding": {"init_A_obsSpots_0": "A"},
                    "featureSelection": {"init_A_obsSpots_0": "A"}
                }
            },
            segmentation_layer_id: {
                "segmentationLayer": {
                    "segmentationChannel": {
                        segmentation_layer_id: segmentation_channels
                    }
                },
                "segmentationChannel": {
                    "obsType": {},
                    "obsColorEncoding": {},
                    "spatialTargetC": {},
                    "spatialChannelVisible": {},
                    "spatialChannelOpacity": {},
                    "spatialChannelColor": {}
                }
            }
        }
    }

    # Add image channels (preserved as requested)
    for ch_id, props in image_channels.items():
        coordination_space["imageChannel"][ch_id] = "__dummy__"
        coordination_space["spatialChannelColor"][ch_id] = props["color"]
        coordination_space["spatialChannelOpacity"][ch_id] = 1.0
        coordination_space["spatialChannelVisible"][ch_id] = True
        coordination_space["spatialChannelWindow"][ch_id] = props["window"]
        coordination_space["spatialTargetC"][ch_id] = props["targetC"]

        for key in ["spatialChannelColor", "spatialChannelOpacity", "spatialChannelVisible", "spatialChannelWindow", "spatialTargetC"]:
            coordination_space["metaCoordinationScopesBy"][image_layer_id]["imageChannel"][key][ch_id] = ch_id

    # Add segmentation channels for ROI interactions
    for idx, (interaction, color) in enumerate(zip(roi_interactions, roi_colors)):
        channel_id = segmentation_channels[idx]
        
        coordination_space["segmentationChannel"][channel_id] = "__dummy__"
        coordination_space["spatialTargetC"][channel_id] = idx
        coordination_space["spatialChannelColor"][channel_id] = color
        coordination_space["spatialChannelOpacity"][channel_id] = 0.7
        coordination_space["spatialChannelVisible"][channel_id] = True

        # Add to meta coordination scopes
        coordination_space["metaCoordinationScopesBy"][segmentation_layer_id]["segmentationChannel"]["obsType"][channel_id] = channel_id
        coordination_space["metaCoordinationScopesBy"][segmentation_layer_id]["segmentationChannel"]["obsColorEncoding"][channel_id] = channel_id
        coordination_space["metaCoordinationScopesBy"][segmentation_layer_id]["segmentationChannel"]["spatialTargetC"][channel_id] = channel_id
        coordination_space["metaCoordinationScopesBy"][segmentation_layer_id]["segmentationChannel"]["spatialChannelVisible"][channel_id] = dataset_uid
        coordination_space["metaCoordinationScopesBy"][segmentation_layer_id]["segmentationChannel"]["spatialChannelOpacity"][channel_id] = dataset_uid
        coordination_space["metaCoordinationScopesBy"][segmentation_layer_id]["segmentationChannel"]["spatialChannelColor"][channel_id] = channel_id

    # Dataset files configuration
    dataset_files = [
        {
            "fileType": "image.ome-zarr",
            "url": "https://lsp-public-data.s3.amazonaws.com/biomedvis-challenge-2025/Dataset1-LSP13626-melanoma-in-situ/0"
        },
        {
            "fileType": "obsSpots.spatialdata.zarr",
            "url": "http://localhost:5000/api/zarr/roi_shapes.spatialdata.zarr",
            "coordinationValues": {
                "obsType": "spot"
            },
            "options": {
                "path": "shapes/roi_circles",
                "tablePath": "table/table"
            }
        }
    ]

    return {
        "version": "1.0.16",
        "name": "BioMedVis Challenge - Image and ROI Segmentation",
        "description": "Render base image with marker channels and ROI segmentation layer using zarr file",
        "initStrategy": "auto",
        "datasets": [{
            "uid": dataset_uid,
            "name": "Blood Vessel",
            "files": dataset_files
        }],
        "coordinationSpace": coordination_space,
        "layout": [
            {
                "component": "spatialBeta",
                "coordinationScopes": {
                    "dataset": dataset_uid,
                    "imageLayer": image_layer_id,
                    "segmentationLayer": segmentation_layer_id,
                    "metaCoordinationScopes": ["init_A_obsSpots_0", image_layer_id, segmentation_layer_id],
                    "metaCoordinationScopesBy": ["init_A_obsSpots_0", image_layer_id, segmentation_layer_id],
                    "spatialSegmentationLayer": ["A", "B", "C", "D"],
                    "obsType": ["A", "B", "C", "D"],
                    "obsColorEncoding": ["A", "B", "C", "D"],
                    "imageChannel": list(image_channels.keys()),
                    "spatialChannelColor": list(image_channels.keys()),
                    "spatialChannelOpacity": list(image_channels.keys()),
                    "spatialChannelVisible": list(image_channels.keys()),
                    "spatialChannelWindow": list(image_channels.keys()),
                    "spatialTargetC": list(image_channels.keys()),
                    "spatialRenderingMode": image_layer_id,
                    "spatialLayerOpacity": image_layer_id,
                    "spatialLayerVisible": image_layer_id,
                    "spatialTargetResolution": image_layer_id,
                    "spatialTargetT": image_layer_id,
                    "spatialTargetX": image_layer_id,
                    "spatialTargetY": image_layer_id,
                    "spatialTargetZ": image_layer_id,
                    "spatialZoom": image_layer_id
                },
                "x": 0, "y": 0, "w": 8, "h": 12
            },
            {
                "component": "layerControllerBeta",
                "coordinationScopes": {
                    "dataset": dataset_uid,
                    "imageLayer": image_layer_id,
                    "segmentationLayer": segmentation_layer_id,
                    "metaCoordinationScopes": ["init_A_obsSpots_0", image_layer_id, segmentation_layer_id],
                    "metaCoordinationScopesBy": ["init_A_obsSpots_0", image_layer_id, segmentation_layer_id],
                    "spatialSegmentationLayer": ["A", "B", "C", "D"],
                    "obsType": ["A", "B", "C", "D"],
                    "imageChannel": list(image_channels.keys()),
                    "spatialChannelColor": list(image_channels.keys()),
                    "spatialChannelOpacity": list(image_channels.keys()),
                    "spatialChannelVisible": list(image_channels.keys()),
                    "spatialChannelWindow": list(image_channels.keys()),
                    "spatialTargetC": list(image_channels.keys()),
                    "spatialRenderingMode": image_layer_id,
                    "spatialLayerOpacity": image_layer_id,
                    "spatialLayerVisible": image_layer_id,
                    "spatialTargetResolution": image_layer_id,
                    "spatialTargetT": image_layer_id,
                    "spatialTargetX": image_layer_id,
                    "spatialTargetY": image_layer_id,
                    "spatialTargetZ": image_layer_id,
                    "spatialZoom": image_layer_id
                },
                "x": 8, "y": 0, "w": 4, "h": 12
            }
        ]
    }


## end config generator













@app.route('/api/config', methods=['GET'])
def get_config():
    """Get the standard Vitessce config using Python generator"""
    logger.info("Request received for /api/config [GET]")
    
    try:
        config = generate_bio_med_vis_config()
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

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000) 