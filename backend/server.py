# Backend server code will go here 
import json
import logging
from flask import Flask, jsonify, request, Response, stream_with_context
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
import zarr # Use zarr directly
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
# Use local Zarr file
ZARR_BASE_URL = None  # We'll use local file
LOCAL_ZARR_PATH = Path(__file__).parent / 'input' / 'selected_channels.zarr'
ZARR_IMAGE_GROUP_PATH = "data"
TARGET_RESOLUTION_PATH = "" # Use the root of data directory
# ---
REQUIRED_CHANNEL_INDICES_FOR_RGB_INTERACTION = {
    'CD31': 0,
    'CD11b': 1,
    'Catalase': 2,
    'CD4': 3,
    'CD20': 4,
    'CD11c': 5
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
    
    # Return default channel mapping for local development
    channel_map = {
        '0': 'CD31',
        '1': 'CD11b', 
        '2': 'Catalase',
        '3': 'CD4',
        '4': 'CD20',
        '5': 'CD11c'
    }
    
    logger.info(f"Returning default channel names: {channel_map}")
    return jsonify(channel_map)
# --- End Channel Names Endpoint ---

def open_target_zarr_array():
    """Opens and returns the target Zarr array from local file. Returns None on failure."""
    target_image_arr = None
    try:
        # Use local Zarr file - open as group first
        zarr_path = LOCAL_ZARR_PATH
        
        logger.info(f"Attempting to open local Zarr group from: {zarr_path}")
        
        # Check if path exists
        if not zarr_path.exists():
            logger.error(f"Zarr path does not exist: {zarr_path}")
            return None
        
        # Open the Zarr group
        zarr_group = zarr.open_group(str(zarr_path), mode='r')
        logger.info(f"Zarr group keys: {list(zarr_group.keys())}")
        
        # Look for the data array - handle nested structure
        if 'data' in zarr_group:
            data_group = zarr_group['data']
            logger.info(f"Found 'data' group. Keys: {list(data_group.keys())}")
            
            # Check if data is directly an array or has nested structure
            if hasattr(data_group, 'shape'):
                # data is directly an array
                target_image_arr = data_group
                logger.info(f"✅ Successfully opened target Zarr array from local file")
                logger.info(f"Array shape: {target_image_arr.shape}")
                logger.info(f"Array dtype: {target_image_arr.dtype}")
                return target_image_arr
            else:
                # data is a group, look for the actual array
                for key in data_group.keys():
                    if hasattr(data_group[key], 'shape'):
                        target_image_arr = data_group[key]
                        logger.info(f"✅ Successfully opened target Zarr array from local file (nested)")
                        logger.info(f"Array shape: {target_image_arr.shape}")
                        logger.info(f"Array dtype: {target_image_arr.dtype}")
                        return target_image_arr
                    elif hasattr(data_group[key], 'keys'):
                        # Check nested groups (like data/c/0)
                        nested_group = data_group[key]
                        logger.info(f"Found nested group '{key}'. Keys: {list(nested_group.keys())}")
                        for nested_key in nested_group.keys():
                            if hasattr(nested_group[nested_key], 'shape'):
                                target_image_arr = nested_group[nested_key]
                                logger.info(f"✅ Successfully opened target Zarr array from nested group '{key}/{nested_key}'")
                                logger.info(f"Array shape: {target_image_arr.shape}")
                                logger.info(f"Array dtype: {target_image_arr.dtype}")
                                return target_image_arr
                            elif hasattr(nested_group[nested_key], 'keys'):
                                # Check deeper nested groups (like data/c/0/0)
                                deep_nested_group = nested_group[nested_key]
                                logger.info(f"Found deep nested group '{key}/{nested_key}'. Keys: {list(deep_nested_group.keys())}")
                                for deep_key in deep_nested_group.keys():
                                    if hasattr(deep_nested_group[deep_key], 'shape'):
                                        target_image_arr = deep_nested_group[deep_key]
                                        logger.info(f"✅ Successfully opened target Zarr array from deep nested group '{key}/{nested_key}/{deep_key}'")
                                        logger.info(f"Array shape: {target_image_arr.shape}")
                                        logger.info(f"Array dtype: {target_image_arr.dtype}")
                                        return target_image_arr
                                    elif hasattr(deep_nested_group[deep_key], 'keys'):
                                        # Check even deeper nested groups (like data/c/0/0/100)
                                        deeper_nested_group = deep_nested_group[deep_key]
                                        logger.info(f"Found deeper nested group '{key}/{nested_key}/{deep_key}'. Keys: {list(deeper_nested_group.keys())}")
                                        # For this level, we'll use the first available key as a sample
                                        if deeper_nested_group.keys():
                                            sample_key = list(deeper_nested_group.keys())[0]
                                            sample_group = deeper_nested_group[sample_key]
                                            if hasattr(sample_group, 'shape'):
                                                target_image_arr = sample_group
                                                logger.info(f"✅ Successfully opened target Zarr array from deeper nested group '{key}/{nested_key}/{deep_key}/{sample_key}'")
                                                logger.info(f"Array shape: {target_image_arr.shape}")
                                                logger.info(f"Array dtype: {target_image_arr.dtype}")
                                                return target_image_arr
                                            elif hasattr(sample_group, 'keys'):
                                                # Check even deeper nested groups (like data/c/0/0/100/0)
                                                even_deeper_nested_group = sample_group
                                                logger.info(f"Found even deeper nested group '{key}/{nested_key}/{deep_key}/{sample_key}'. Keys: {list(even_deeper_nested_group.keys())}")
                                                # For this level, we'll use the first available key as a sample
                                                if even_deeper_nested_group.keys():
                                                    even_sample_key = list(even_deeper_nested_group.keys())[0]
                                                    even_sample_group = even_deeper_nested_group[even_sample_key]
                                                    if hasattr(even_sample_group, 'shape'):
                                                        target_image_arr = even_sample_group
                                                        logger.info(f"✅ Successfully opened target Zarr array from even deeper nested group '{key}/{nested_key}/{deep_key}/{sample_key}/{even_sample_key}'")
                                                        logger.info(f"Array shape: {target_image_arr.shape}")
                                                        logger.info(f"Array dtype: {target_image_arr.dtype}")
                                                        return target_image_arr
                
                logger.error(f"No array found in 'data' group. Available keys: {list(data_group.keys())}")
                return None
        else:
            logger.error(f"No 'data' group found in Zarr group. Available keys: {list(zarr_group.keys())}")
            return None

    except Exception as e:
        logger.error(f"Failed to open target Zarr array: {e}", exc_info=True)
        return None

def calculate_z_projection_heatmap(zarr_array, channel_index, roi_z, roi_y, roi_x, projection_type='sum'):
    """Calculates the Z-projection (heatmap) for a specific channel within a defined ROI."""
    if zarr_array is None:
        return {'error': 'Zarr array is None.'}
    valid_projection_types = ['sum', 'mean', 'max']
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
        
        # Use ROI Z range for projection instead of full Z range
        roi_slice = (
            slice(0, 1),
            slice(channel_index, channel_index + 1),
            slice(roi_z[0], roi_z[1]),  # Use ROI Z range instead of full Z range
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
            
            if projection_type == 'sum':
                # Additive projection (sum along Z axis)
                projected_data = np.sum(data_roi, axis=2)
            elif projection_type == 'mean':
                # Mean projection along Z axis
                projected_data = np.mean(data_roi, axis=2)
            elif projection_type == 'max':
                # Max projection along Z axis
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

def calculate_rgb_interaction_heatmap_py(zarr_array, roi_z, roi_y, roi_x, projection_type='sum'):
    if zarr_array is None:
        return {'error': 'Zarr array is None.'}
    valid_projection_types = ['sum', 'mean', 'max']
    if projection_type not in valid_projection_types:
        return {'error': f"Invalid projection_type '{projection_type}'. Must be one of {valid_projection_types}"}
    try:
        if zarr_array.ndim != 5:
            logger.warning(f"[Interaction Heatmap] Expected 5D array (t,c,z,y,x), but got {zarr_array.ndim}D. Proceeding cautiously.")
        
        # Define channel groups and their ranges - updated for new interactions
        channel_groups = {
            1: [4, 0],       # group 1: CD20 + CD31 (B-cell infiltration)
            2: [3, 0],       # group 2: CD4 + CD31 (T-cell maturation)
            3: [1, 5],       # group 3: CD11b + CD11c (Inflammatory zone)
            4: [1, 2]        # group 4: CD11b + Catalase (Oxidative stress regulation)
        }
        
        # Define channel ranges
        channel_ranges = {
            0: [300, 20000],     # CD31
            4: [1000, 7000],     # CD20
            1: [700, 6000],      # CD11b
            3: [1638, 10000],    # CD4
            2: [1638, 7000],     # Catalase
            5: [370, 1432]       # CD11c
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
                
                # Step 1: Get the maximum value for this channel from the ROI only
                roi_channel_slice = (
                    slice(0, 1),
                    slice(ch_idx, ch_idx + 1),
                    slice(roi_z[0], roi_z[1]),  # Use ROI Z range
                    slice(zarr_slice_y_start, zarr_slice_y_end),
                    slice(roi_x[0], roi_x[1])
                )
                roi_channel_data = zarr_array[roi_channel_slice]
                
                # Apply channel-specific range filtering to ROI data
                if ch_idx in channel_ranges:
                    min_range, max_range = channel_ranges[ch_idx]
                    roi_channel_data = np.clip(roi_channel_data, min_range, max_range)
                
                # Get the maximum value from the ROI for this channel
                roi_max_val = np.max(roi_channel_data)
                logger.info(f"  Channel {ch_idx} ROI max value: {roi_max_val:.3f}")
                
                # Use the same ROI data for processing
                data_roi = roi_channel_data
                
                # Step 2: Normalize ROI data using the ROI maximum for this channel
                data_roi = data_roi / roi_max_val
                
                # Step 3: Calculate summation over Z for each x,y position
                if projection_type == 'sum':
                    heatmap_2d = np.sum(data_roi, axis=2)
                elif projection_type == 'mean':
                    heatmap_2d = np.mean(data_roi, axis=2)
                elif projection_type == 'max':
                    heatmap_2d = np.max(data_roi, axis=2)
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
            
            # Step 4: Calculate interaction as summation of normalized channels for each x,y
            # Stack all channel heatmaps
            stacked_heatmaps = np.stack(channel_heatmaps)
            
            # Calculate summation along channel axis (axis=0) for each x,y position
            interaction = np.sum(stacked_heatmaps, axis=0)
            
            # Step 5: Final normalization of interaction values to [0, 1] range
            min_val = np.min(interaction)
            max_val = np.max(interaction)
            logger.info(f"\nGroup {group_val} interaction before normalization:")
            logger.info(f"  Min value: {min_val:.6f}")
            logger.info(f"  Max value: {max_val:.6f}")
            logger.info(f"  Range: {max_val - min_val:.6f}")
            
            if max_val > min_val:
                # Simple min-max normalization to scale values between 0 and 1
                # This ensures the minimum value becomes 0 and maximum value becomes 1
                interaction = (interaction - min_val) / (max_val - min_val)
                
                logger.info(f"  After min-max normalization:")
                logger.info(f"    Min value: {np.min(interaction):.6f} (was {min_val:.8f})")
                logger.info(f"    Max value: {np.max(interaction):.6f} (was {max_val:.8f})")
                logger.info(f"    Mean value: {np.mean(interaction):.6f}")
            else:
                interaction = np.zeros_like(interaction)
                logger.info(f"  All values are the same, setting to zero")
            
            # Log statistics about the interaction values
            non_zero_mask = interaction > 0
            if np.any(non_zero_mask):
                non_zero_values = interaction[non_zero_mask]
                logger.info(f"\nGroup {group_val} interaction statistics:")
                logger.info(f"  Channel stats:")
                for i, stats in enumerate(channel_stats):
                    logger.info(f"    Channel {channels[i]}:")
                    logger.info(f"      Max: {stats['max']:.6f}")
                    logger.info(f"      Mean: {stats['mean']:.6f}")
                    logger.info(f"      Non-zero: {stats['non_zero']}")
                logger.info(f"  Interaction stats:")
                logger.info(f"    Total pixels: {interaction.size}")
                logger.info(f"    Non-zero pixels: {np.sum(non_zero_mask)}")
                logger.info(f"    Values near 1 (>0.9): {np.sum(non_zero_values > 0.9)}")
                logger.info(f"    Values near 0 (<0.1): {np.sum(non_zero_values < 0.1)}")
                logger.info(f"    Mean value: {np.mean(non_zero_values):.6f}")
                logger.info(f"    Median value: {np.median(non_zero_values):.6f}")
                logger.info(f"    Max value: {np.max(non_zero_values):.6f}")
                logger.info(f"    Min value: {np.min(non_zero_values):.6f}")
                
                # Print to terminal for debugging
                print(f"\n=== INTERACTION GROUP {group_val} DEBUG ===")
                print(f"Max value: {np.max(interaction):.8f}")
                print(f"Min value: {np.min(interaction):.8f}")
                print(f"Mean value: {np.mean(interaction):.8f}")
                print(f"95th percentile: {np.percentile(interaction, 95):.8f}")
                print(f"99th percentile: {np.percentile(interaction, 99):.8f}")
                print(f"Non-zero pixels: {np.sum(non_zero_mask)} / {interaction.size}")
                print(f"==========================================\n")
            
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

@app.route('/api/top_roi_scores_<interaction_type>', methods=['GET'])
def serve_top_roi_scores(interaction_type):
    """Serve top ROI scores for a specific interaction type"""
    logger.info(f"Request received for top ROI scores with interaction_type: '{interaction_type}'")
    try:
        # Decode the interaction type and convert to filename format
        from urllib.parse import unquote
        decoded_interaction_type = unquote(interaction_type)
        filename = f"all_roi_scores_{decoded_interaction_type}.json"
        logger.info(f"Looking for file: {filename}")
        roi_path = Path(__file__).parent / "output" / filename
        
        if not roi_path.exists():
            logger.warning(f"Top ROI scores file not found: {filename}")
            return jsonify({"error": f"Top ROI scores file not found: {filename}"}), 404

        with open(roi_path, 'r') as f:
            roi_data = json.load(f)
            
        return jsonify(roi_data)
    except Exception as e:
        logger.error(f"Error serving top ROI scores for {interaction_type}: {e}", exc_info=True)
        return jsonify({"error": f"Failed to serve top ROI scores for {interaction_type}: {e}"}), 500

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
        
        roi_info = data.get('roiInfo', {})
        
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
            'CD31': 0,
            'CD11b': 1,
            'Catalase': 2,
            'CD4': 3,
            'CD20': 4,
            'CD11c': 5
        }
        
        # Process only specified channels
        heatmaps = {}
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
                    projection_type='sum'
                )
                
                if 'error' not in heatmap_result:
                    heatmaps[channel_name] = {
                        "data": heatmap_result.get('heatmap', []),
                        "shape": heatmap_result.get('shape'),
                        "range": {
                            "min": heatmap_result.get('min_val', 0),
                            "max": heatmap_result.get('max_val', 1),
                            "mean": np.mean(heatmap_result.get('heatmap', [])) if heatmap_result.get('heatmap') else 0
                        }
                    }
                    logger.info(f"Successfully processed channel {channel_name}")
                else:
                    logger.error(f"Error processing channel {channel_name}: {heatmap_result['error']}")
            except Exception as channel_error:
                logger.error(f"Error processing channel {channel_name}: {str(channel_error)}", exc_info=True)
                continue
        
        if not heatmaps:
            logger.error("No results were generated for any channel")
            return jsonify({'error': 'Failed to generate heatmaps for any channel'}), 500
        
        # Calculate overall statistics
        all_values = []
        for channel_data in heatmaps.values():
            if isinstance(channel_data['data'], list):
                all_values.extend([val for row in channel_data['data'] for val in row])
        # what happend here why we have hardcode for value
        statistics = {
            "mean_intensity": np.mean(all_values) if all_values else 0.5,
            "max_intensity": np.max(all_values) if all_values else 1.0,
            "min_intensity": np.min(all_values) if all_values else 0.1,
            "total_channels": len(heatmaps),
            "available_channels": list(heatmaps.keys())
        }
            
        logger.info(f"Returning results for {len(heatmaps)} channels")
        
        heatmap_data = {
            "roi_info": roi_info,
            "roi_bounds": roi,
            "channels": channels,
            "heatmaps": heatmaps,
            "statistics": statistics
        }
        
        return jsonify(heatmap_data)
        
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
            'sum'
        )
        
        if 'error' in result:
            logger.error(f"Error in interaction heatmap calculation: {result['error']}")
            return jsonify(result), 400
            
        logger.info("Successfully calculated interaction heatmaps")
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error in analyze_interaction_heatmap: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500



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

@app.route('/api/roi_segmentation_T-cell_maturation.json', methods=['GET'])
def get_roi_segmentation_t_cell():
    """Serve the roi_segmentation_T-cell_maturation.json file"""
    logger.info("Request received for /api/roi_segmentation_T-cell_maturation.json [GET]")
    
    try:
        roi_file_path = Path(__file__).parent / 'output' / 'roi_segmentation_T-cell_maturation.json'
        
        if not roi_file_path.exists():
            logger.error(f"roi_segmentation_T-cell_maturation.json file not found: {roi_file_path}")
            return jsonify({"error": "roi_segmentation_T-cell_maturation.json file not found"}), 404
        
        with open(roi_file_path, 'r') as f:
            roi_data = json.load(f)
        
        return jsonify(roi_data)
        
    except Exception as e:
        logger.error(f"Error serving roi_segmentation_T-cell_maturation.json: {e}", exc_info=True)
        return jsonify({"error": f"Failed to serve roi_segmentation_T-cell_maturation.json: {e}"}), 500

@app.route('/api/roi_segmentation_Oxidative_stress_regulation.json', methods=['GET'])
def get_roi_segmentation_oxidative():
    """Serve the roi_segmentation_Oxidative_stress_regulation.json file"""
    logger.info("Request received for /api/roi_segmentation_Oxidative_stress_regulation.json [GET]")
    
    try:
        roi_file_path = Path(__file__).parent / 'output' / 'roi_segmentation_Oxidative_stress_regulation.json'
        
        if not roi_file_path.exists():
            logger.error(f"roi_segmentation_Oxidative_stress_regulation.json file not found: {roi_file_path}")
            return jsonify({"error": "roi_segmentation_Oxidative_stress_regulation.json file not found"}), 404
        
        with open(roi_file_path, 'r') as f:
            roi_data = json.load(f)
        
        return jsonify(roi_data)
        
    except Exception as e:
        logger.error(f"Error serving roi_segmentation_Oxidative_stress_regulation.json: {e}", exc_info=True)
        return jsonify({"error": f"Failed to serve roi_segmentation_Oxidative_stress_regulation.json: {e}"}), 500

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
                    
                    # Add tooltip_name for better tooltip display
                    if 'interaction' in feature['properties'] and 'score' in feature['properties']:
                        interaction = feature['properties']['interaction']
                        roi_id = feature['properties'].get('id', 'Unknown')
                        score = feature['properties']['score']
                        feature['properties']['tooltip_name'] = f"{interaction}_{roi_id}_Score:{score:.3f}"
            
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
    