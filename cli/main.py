# In cli/main.py

import argparse
import cv2
import sys
import os
import matplotlib.pyplot as plt

# This is a crucial step to allow main.py to find the 'core' module
# It adds the parent directory of 'cli' (which is the project root 'ASTROLITH') to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Now we can import from our custom modules
from core import utils
from core import pose_estimator

def parse_args():
    """Parses command-line arguments for the main application."""
    parser = argparse.ArgumentParser(
        description="Main application for detecting ArUco markers and calculating box pose."
    )
    # Note: Paths are now relative to the project root (e.g., 'assets/demo/image.jpeg')
    parser.add_argument('--profile', type=str, required=True, help="Path to the camera calibration profile file (e.g., 'assets/profiles/cam.npz').")
    parser.add_argument('--image', type=str, required=True, help="Path to the input image file (e.g., 'assets/demo/box.jpeg').")
    parser.add_argument('--marker-size', type=float, required=True, choices=[5, 10, 15, 20],
                        help='The side length of the markers in centimeters. Must be one of: 5, 10, 15, 20.')
    parser.add_argument('--aruco_dict', type=str, default='DICT_5X5_100', help='The ArUco dictionary to use.')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 1. Load Data using helper functions from the 'utils' module
    camera_matrix, dist_coeffs = utils.load_calibration_profile(args.profile)
    if camera_matrix is None: sys.exit(1)
    
    aruco_dict = utils.get_aruco_dictionary(args.aruco_dict)
    if aruco_dict is None: sys.exit(1)
    
    frame = cv2.imread(args.image)
    if frame is None:
        print(f"Error: Could not read image file at '{args.image}'")
        return
        
    marker_size_meters = args.marker_size / 100.0

    # 2. Process the image using the core engine from 'pose_estimator' module
    results, visualized_frame = pose_estimator.find_box_pose(
        frame, camera_matrix, dist_coeffs, aruco_dict, marker_size_meters
    )

    # 3. Print and display the results
    if results:
        print("\n" + "="*50)
        print("FINAL BOX DIMENSIONING DATA (in meters, relative to camera)")
        print("="*50)
        print(f"\n[1] Common Origin (Box Corner):\n    {results['common_origin']}")
        print(f"\n[2] Box Axes (Direction Vectors):")
        print(f"    - Length Axis: {results['length_axis']}")
        print(f"    - Width Axis:  {results['width_axis']}")
        print(f"    - Height Axis: {results['height_axis']}")
        print("="*50)

    # Display the final image using Matplotlib
    print("\nDisplaying final result...")
    rgb_frame = cv2.cvtColor(visualized_frame, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(12, 8))
    plt.imshow(rgb_frame)
    plt.title('ArUco Detection Result')
    plt.axis('off')
    plt.show()

    print("Program finished.")

if __name__ == '__main__':
    main()