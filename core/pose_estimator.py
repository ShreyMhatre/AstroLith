import cv2
import numpy as np
from collections import Counter

def find_line_intersection(line1_p1, line1_p2, line2_p1, line2_p2):
    """
    Finds the intersection of two lines defined by two points each.
    Returns the intersection point (x, y) or None if lines are parallel.
    """
    x1, y1 = line1_p1; x2, y2 = line1_p2
    x3, y3 = line2_p1; x4, y4 = line2_p2

    denominator = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if denominator == 0:
        return None # Lines are parallel

    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denominator
    intersection_x = x1 + t * (x2 - x1)
    intersection_y = y1 + t * (y2 - y1)
    
    return np.array([intersection_x, intersection_y], dtype=np.float32)

def refine_corners_subpixel_advanced(gray, corners):
    """
    Applies advanced subpixel corner refinement with edge detection.
    This significantly improves accuracy for measurement applications.
    """
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.00001)
    
    refined_corners_list = []
    for marker_corners_array in corners:
        # Multiple passes for refinement
        refined1 = cv2.cornerSubPix(gray, marker_corners_array.copy(), (7, 7), (-1,-1), criteria)
        refined2 = cv2.cornerSubPix(gray, refined1.copy(), (5, 5), (-1, -1), criteria)
        refined_corners_list.append(refined2)
    
    return refined_corners_list

def find_box_pose(frame, camera_matrix, dist_coeffs, aruco_dict, marker_size_meters):
    """
    Detects markers with high-accuracy corner refinement, calculates pose,
    and returns the 3D data.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    objp_marker15 = np.array([[0, marker_size_meters, 0], [marker_size_meters, marker_size_meters, 0], [marker_size_meters, 0, 0], [0, 0, 0]], dtype=np.float32)
    objp_marker5 = np.array([[-marker_size_meters, marker_size_meters, 0], [0, marker_size_meters, 0], [0, 0, 0], [-marker_size_meters, 0, 0]], dtype=np.float32)

    # --- HIGH-ACCURACY DETECTION PARAMETERS ---
    aruco_params = cv2.aruco.DetectorParameters()
    aruco_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_CONTOUR
    aruco_params.cornerRefinementWinSize = 7
    aruco_params.cornerRefinementMaxIterations = 50
    aruco_params.cornerRefinementMinAccuracy = 0.005
    aruco_params.adaptiveThreshWinSizeMin = 3
    aruco_params.adaptiveThreshWinSizeMax = 25
    aruco_params.adaptiveThreshWinSizeStep = 4
    aruco_params.adaptiveThreshConstant = 5
    aruco_params.polygonalApproxAccuracyRate = 0.02
    aruco_params.minCornerDistanceRate = 0.03
    aruco_params.perspectiveRemovePixelPerCell = 8
    aruco_params.errorCorrectionRate = 0.8
    
    corners, ids, rejected = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=aruco_params)
    
    pose_data = {}
    corners_data = {}
    
    if ids is not None:
        # --- STEP 1: ADVANCED SUBPIXEL REFINEMENT ---
        print("Applying advanced corner refinement...")
        refined_corners_list = refine_corners_subpixel_advanced(gray, corners)
        
        # --- STEP 2: LINE INTERSECTION REFINEMENT ---
        print("Applying line intersection refinement...")
        final_corners_list = []
        for marker_corners_array in refined_corners_list:
            p = marker_corners_array[0]
            tl, tr, br, bl = p[0], p[1], p[2], p[3]
            
            new_tl = find_line_intersection(tl, tr, tl, bl)
            new_tr = find_line_intersection(tr, tl, tr, br)
            new_br = find_line_intersection(br, tr, br, bl)
            new_bl = find_line_intersection(bl, br, bl, tl)
            
            if all(c is not None for c in [new_tl, new_tr, new_br, new_bl]):
                new_corners = np.array([[new_tl, new_tr, new_br, new_bl]], dtype=np.float32)
                final_corners_list.append(new_corners)
            else:
                final_corners_list.append(marker_corners_array) # Fallback
        
        corners = tuple(final_corners_list)

        for i, marker_id_array in enumerate(ids):
            marker_id = marker_id_array[0]
            if marker_id == 5: obj_points = objp_marker5
            elif marker_id == 15: obj_points = objp_marker15
            else: continue

            success, rvec, tvec = cv2.solvePnP(obj_points, corners[i], camera_matrix, dist_coeffs)
            if success:
                pose_data[marker_id] = {'rvec': rvec, 'tvec': tvec}
                corners_data[marker_id] = corners[i]
    else:
        print("No markers detected.")
        return None

    if 5 in pose_data and 15 in pose_data:
        rvec5, tvec5 = pose_data[5]['rvec'], pose_data[5]['tvec']
        rvec15, tvec15 = pose_data[15]['rvec'], pose_data[15]['tvec']
        rot_mat5, _ = cv2.Rodrigues(rvec5)
        rot_mat15, _ = cv2.Rodrigues(rvec15)

        common_origin = (tvec5.flatten() + tvec15.flatten()) / 2.0
        
        # Corrected width axis direction
        width_axis = -rot_mat5[:, 0]
        length_axis = rot_mat15[:, 0]
        
        height_axis_5 = rot_mat5[:, 1]
        height_axis_15 = rot_mat15[:, 1]
        avg_height_axis = (height_axis_5 + height_axis_15) / 2.0
        height_axis = avg_height_axis / np.linalg.norm(avg_height_axis)

        results = {
            "common_origin": common_origin,
            "length_axis": length_axis,
            "width_axis": width_axis,
            "height_axis": height_axis,
            "corners_data": corners_data,
            "pose_data": pose_data
        }
        
        print("\nPose estimation successful!")
        return results
    else:
        print("Error: Could not find both markers 5 and 15.")
        return None