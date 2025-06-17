# src/parkingspace/regions.py

import json
import numpy as np

def save_regions_to_file(
    upper_level_l, upper_level_m, upper_level_r,
    close_perp, far_side, close_side, far_perp, small_park,
    ignore_regions, file_path='regions.json'
):
    # Save parking regions to JSON file
    data = {
        'upper_level_l': upper_level_l.tolist(),
        'upper_level_m': upper_level_m.tolist(),
        'upper_level_r': upper_level_r.tolist(),
        'close_perp': close_perp.tolist(),
        'far_side': far_side.tolist(),
        'close_side': close_side.tolist(),
        'far_perp': far_perp.tolist(),
        'small_park': small_park.tolist(),
        'ignore_regions': [region.tolist() for region in ignore_regions]
    }
    
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=2)

def load_regions_from_file(file_path='regions.json'):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return (
        np.array(data['upper_level_l']),
        np.array(data['upper_level_m']),
        np.array(data['upper_level_r']),
        np.array(data['close_perp']),
        np.array(data['far_side']),
        np.array(data['close_side']),
        np.array(data['far_perp']),
        np.array(data['small_park']),
        [np.array(region) for region in data['ignore_regions']]
    )

def get_thresholds():
    # The same dictionary you used before:
    return {
        'upper_level_l': {
            'min_area': 2000,
            'max_aspect_ratio': 16,
            'min_solidity': 0.7,
            'min_width': 60,
            'max_width': 500,
            'min_height': 40,
            'max_height': 300
        },
        'upper_level_m': {
            'min_area': 2000,
            'max_aspect_ratio': 16,
            'min_solidity': 0.7,
            'min_width': 120,
            'max_width': 1050,
            'min_height': 50,
            'max_height': 300
        },
        'upper_level_r': {
            'min_area': 2000,
            'max_aspect_ratio': 16,
            'min_solidity': 0.7,
            'min_width': 170,
            'max_width': 500,
            'min_height': 100,
            'max_height': 150
        },
        'close_perp': {
            'min_area': 10,
            'max_aspect_ratio': 5,
            'min_solidity': 0.6,
            'min_width': 10,
            'max_width': 200,
            'min_height': 10,
            'max_height': 200
        },
        'far_side': {
            'min_area': 100,
            'max_aspect_ratio': 5,
            'min_solidity': 0.7,
            'min_width': 30,
            'max_width': 200,
            'min_height': 30,
            'max_height': 200
        },
        'close_side': {
            'min_area': 100,
            'max_aspect_ratio': 5,
            'min_solidity': 0.6,
            'min_width': 30,
            'max_width': 200,
            'min_height': 30,
            'max_height': 200
        },
        'far_perp': {
            'min_area': 100,
            'max_aspect_ratio': 5,
            'min_solidity': 0.7,
            'min_width': 30,
            'max_width': 200,
            'min_height': 30,
            'max_height': 200
        },
        'small_park': {
            'min_area': 200,
            'max_aspect_ratio': 2,
            'min_solidity': 0.9,
            'min_width': 30,
            'max_width': 200,
            'min_height': 30,
            'max_height': 200
        }
    }
