# In core/utils.py

import cv2
import numpy as np
import os

def load_calibration_profile(profile_path):
    """
    Loads camera matrix and distortion coefficients from a .npz file.
    """
    if not os.path.exists(profile_path):
        print(f"Error: Calibration profile not found at {profile_path}")
        return None, None
    try:
        data = np.load(profile_path)
        return data['camera_matrix'], data['dist_coeffs']
    except KeyError:
        print(f"Error: File {profile_path} lacks 'camera_matrix' or 'dist_coeffs'.")
        return None, None

def get_aruco_dictionary(dict_name):
    """
    Gets the predefined ArUco dictionary from OpenCV based on its string name.
    """
    if hasattr(cv2.aruco, dict_name):
        return cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dict_name))
    else:
        print(f"Error: ArUco dictionary '{dict_name}' is not supported.")
        return None