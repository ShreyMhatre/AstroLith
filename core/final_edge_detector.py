# In core/final_edge_detector.py

import argparse
import cv2
import numpy as np
import os
import sys
import matplotlib.pyplot as plt

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import utils
from core import pose_estimator

def parse_args():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Detects the primary edges and corners of a box using ArUco pose and Hough Line Transform."
    )
    parser.add_argument('--profile', type=str, required=True, help="Path to the camera calibration profile.")
    parser.add_argument('--image', type=str, required=True, help="Path to the input image.")
    parser.add_argument('--marker-size', type=float, required=True, choices=[5, 10, 15, 20], help='Marker size in cm.')
    parser.add_argument('--aruco_dict', type=str, default='DICT_5X5_100', help='The ArUco dictionary to use.')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 1. Load Data and get the box's 3D pose
    camera_matrix, dist_coeffs = utils.load_calibration_profile(args.profile)
    if camera_matrix is None: sys.exit(1)
    
    aruco_dict = utils.get_aruco_dictionary(args.aruco_dict)
    if aruco_dict is None: sys.exit(1)
    
    frame = cv2.imread(args.image)
    if frame is None:
        print(f"Error: Could not read image file at '{args.image}'")
        return
        
    marker_size_meters = args.marker_size / 100.0

    # The pose_estimator gives us our 3D coordinate system and the initial marker visualization
    results, initial_visualization = pose_estimator.find_box_pose(
        frame.copy(), camera_matrix, dist_coeffs, aruco_dict, marker_size_meters
    )

    if not results:
        print("Could not determine box pose. Cannot proceed with edge detection.")
        # Display the initial visualization even on failure, to help debug
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        plt.imshow(rgb_frame)
        plt.title('ArUco Detection (Pose Failed)')
        plt.axis('off')
        plt.show()
        return

    # 2. Prepare for Edge Detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Use Canny to find all potential edges
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    
    # 3. Project 3D axes to get the expected 2D directions of the box edges
    common_origin = results['common_origin']
    length_axis = results['length_axis']
    width_axis = results['width_axis']
    height_axis = results['height_axis']

    axis_length_3d = 1.0 # Project a long line to get a clean direction
    points_3d = np.array([
        common_origin, 
        common_origin + length_axis * axis_length_3d,
        common_origin + width_axis * axis_length_3d,
        common_origin + height_axis * axis_length_3d
    ])
    points_2d, _ = cv2.projectPoints(points_3d, np.zeros(3), np.zeros(3), camera_matrix, dist_coeffs)
    origin_2d, length_end_2d, width_end_2d, height_end_2d = [p.ravel() for p in points_2d]

    # Normalize the 2D direction vectors
    length_dir_2d = (length_end_2d - origin_2d) / np.linalg.norm(length_end_2d - origin_2d)
    width_dir_2d = (width_end_2d - origin_2d) / np.linalg.norm(width_end_2d - origin_2d)
    height_dir_2d = (height_end_2d - origin_2d) / np.linalg.norm(height_end_2d - origin_2d)

    # 4. Find all straight lines in the image using Hough Line Transform
    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi / 180, threshold=40, minLineLength=50, maxLineGap=15)
    
    if lines is None:
        print("Could not detect any lines in the image.")
        return
        
    best_lines = {'length': None, 'width': None, 'height': None}
    best_scores = {'length': -1, 'width': -1, 'height': -1}

    # 5. Filter the detected lines to find the three that best match our box axes
    for line in lines:
        x1, y1, x2, y2 = line[0]
        
        # Ensure the line is reasonably close to the ArUco origin to filter out background lines
        dist_to_origin = min(np.linalg.norm(np.array([x1, y1]) - origin_2d), 
                             np.linalg.norm(np.array([x2, y2]) - origin_2d))
        
        if dist_to_origin > 200: # Ignore lines that are far away from our area of interest
            continue
            
        line_vec = np.array([x2 - x1, y2 - y1])
        # Avoid division by zero for very short detected lines
        if np.linalg.norm(line_vec) < 1e-6: continue
        line_vec_norm = line_vec / np.linalg.norm(line_vec)
        
        dot_len = abs(np.dot(line_vec_norm, length_dir_2d))
        dot_wid = abs(np.dot(line_vec_norm, width_dir_2d))
        dot_hgt = abs(np.dot(line_vec_norm, height_dir_2d))
        
        if dot_len > best_scores['length']:
            best_scores['length'] = dot_len
            best_lines['length'] = line[0]
        if dot_wid > best_scores['width']:
            best_scores['width'] = dot_wid
            best_lines['width'] = line[0]
        if dot_hgt > best_scores['height']:
            best_scores['height'] = dot_hgt
            best_lines['height'] = line[0]
            
    # 6. Draw the final visualization
    final_frame = initial_visualization
    edge_color = (0, 165, 255) # Orange for the detected edges
    corner_color = (0, 0, 255) # Red for the corners
    origin_2d_int = tuple(origin_2d.astype(int))

    def find_far_endpoint(origin, line):
        """Finds which of the two line endpoints is further from the origin."""
        p1 = np.array([line[0], line[1]])
        p2 = np.array([line[2], line[3]])
        if np.linalg.norm(p1 - origin) > np.linalg.norm(p2 - origin):
            return tuple(p1.astype(int))
        return tuple(p2.astype(int))

    print("\nDetected Box Edges and Corners:")
    if best_lines['length'] is not None:
        far_corner = find_far_endpoint(origin_2d, best_lines['length'])
        cv2.line(final_frame, origin_2d_int, far_corner, edge_color, 3)
        cv2.circle(final_frame, far_corner, 8, corner_color, -1)
        print("  - Length edge and corner FOUND.")
    else: print("  - Length edge NOT FOUND.")

    if best_lines['width'] is not None:
        far_corner = find_far_endpoint(origin_2d, best_lines['width'])
        cv2.line(final_frame, origin_2d_int, far_corner, edge_color, 3)
        cv2.circle(final_frame, far_corner, 8, corner_color, -1)
        print("  - Width edge and corner FOUND.")
    else: print("  - Width edge NOT FOUND.")

    if best_lines['height'] is not None:
        far_corner = find_far_endpoint(origin_2d, best_lines['height'])
        cv2.line(final_frame, origin_2d_int, far_corner, edge_color, 3)
        cv2.circle(final_frame, far_corner, 8, corner_color, -1)
        print("  - Height edge and corner FOUND.")
    else: print("  - Height edge NOT FOUND.")
        
    # Display the final image
    print("\nDisplaying final edge detection result...")
    rgb_frame = cv2.cvtColor(final_frame, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(12, 8))
    plt.imshow(rgb_frame)
    plt.title('Final Box Edge and Corner Detection')
    plt.axis('off')
    plt.show()

    print("Program finished.")

if __name__ == '__main__':
    main()