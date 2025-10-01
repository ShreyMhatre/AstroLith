# In core/pose_estimator.py

import cv2
import numpy as np
from collections import Counter

def find_line_intersection(line1_p1, line1_p2, line2_p1, line2_p2):
    """
    Finds the intersection of two lines defined by two points each.
    Returns the intersection point (x, y) or None if lines are parallel.
    """
    x1, y1 = line1_p1
    x2, y2 = line1_p2
    x3, y3 = line2_p1
    x4, y4 = line2_p2

    denominator = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if denominator == 0:
        return None # Lines are parallel

    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denominator
    intersection_x = x1 + t * (x2 - x1)
    intersection_y = y1 + t * (y2 - y1)
    
    return np.array([intersection_x, intersection_y], dtype=np.float32)

def find_box_pose(frame, camera_matrix, dist_coeffs, aruco_dict, marker_size_meters):
    """
    Detects markers, refines corners via line intersection, calculates pose,
    and returns the 3D data.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    objp_marker15 = np.array([[0, marker_size_meters, 0], [marker_size_meters, marker_size_meters, 0], [marker_size_meters, 0, 0], [0, 0, 0]], dtype=np.float32)
    objp_marker5 = np.array([[-marker_size_meters, marker_size_meters, 0], [0, marker_size_meters, 0], [0, 0, 0], [-marker_size_meters, 0, 0]], dtype=np.float32)

    aruco_params = cv2.aruco.DetectorParameters()
    aruco_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_CONTOUR
    corners, ids, rejected = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=aruco_params)
    
    pose_data = {}
    corners_data = {}
    if ids is not None:
        # --- ACCURACY UPGRADE: REFINE CORNERS VIA LINE INTERSECTION ---
        refined_corners_list = []
        for marker_corners_array in corners:
            # Get the initial corners
            p = marker_corners_array[0]
            tl, tr, br, bl = p[0], p[1], p[2], p[3]
            
            # Calculate the precise intersection of the edge lines
            new_tl = find_line_intersection(tl, tr, tl, bl)
            new_tr = find_line_intersection(tr, tl, tr, br)
            new_br = find_line_intersection(br, tr, br, bl)
            new_bl = find_line_intersection(bl, br, bl, tl)
            
            # Assemble the new, highly accurate corner array
            if all(c is not None for c in [new_tl, new_tr, new_br, new_bl]):
                new_corners = np.array([[new_tl, new_tr, new_br, new_bl]], dtype=np.float32)
                refined_corners_list.append(new_corners)
            else:
                # Fallback to original corners if something went wrong (e.g., parallel lines detected)
                refined_corners_list.append(marker_corners_array)
        
        # Use the NEW, more accurate corners for all further steps
        corners = tuple(refined_corners_list)
        # -----------------------------------------------------------------

        ids_list = ids.flatten().tolist()
        id_counts = Counter(ids_list)

        if id_counts.get(5, 0) > 1 or id_counts.get(15, 0) > 1:
            return None

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
        return None

    if 5 in pose_data and 15 in pose_data:
        tvec5, tvec15 = pose_data[5]['tvec'], pose_data[15]['tvec']
        common_origin = ((tvec5 + tvec15) / 2.0).flatten()
        
        rvec5, rvec15 = pose_data[5]['rvec'], pose_data[15]['rvec']
        rot_mat5, _ = cv2.Rodrigues(rvec5)
        rot_mat15, _ = cv2.Rodrigues(rvec15)
        width_axis = rot_mat5[:, 0]
        length_axis = rot_mat15[:, 0]
        height_axis_5, height_axis_15 = rot_mat5[:, 1], rot_mat15[:, 1]
        avg_height_axis = (height_axis_5 + height_axis_15) / 2.0
        height_axis = avg_height_axis / np.linalg.norm(avg_height_axis)
        
        results = {
            "common_origin": common_origin,
            "length_axis": length_axis,
            "width_axis": width_axis,
            "height_axis": height_axis,
            "corners_data": corners_data
        }
        return results

    return None
