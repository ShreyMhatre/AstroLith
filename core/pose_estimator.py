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

def refine_corners_subpixel_advanced(gray, corners):
    """
    Applies advanced subpixel corner refinement with edge detection.
    This significantly improves accuracy for measurement applications.
    """
    # Apply bilateral filter to preserve edges while reducing noise
    gray_filtered = cv2.bilateralFilter(gray, 9, 75, 75)
    
    # Multiple passes of subpixel refinement
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.00001)
    
    refined_corners_list = []
    for marker_corners_array in corners:
        # First pass: standard subpixel on filtered image
        refined1 = cv2.cornerSubPix(
            gray_filtered,
            marker_corners_array.copy(),
            winSize=(7, 7),
            zeroZone=(-1, -1),
            criteria=criteria
        )
        
        # Second pass: on original gray image
        refined2 = cv2.cornerSubPix(
            gray,
            refined1.copy(),
            winSize=(5, 5),
            zeroZone=(-1, -1),
            criteria=criteria
        )
        
        # Edge-based refinement for maximum accuracy
        refined_corner = refined2.copy()
        for j, point in enumerate(refined2[0]):
            x, y = int(point[0]), int(point[1])
            
            # Extract local patch around corner
            patch_size = 20
            y_start = max(0, y - patch_size)
            y_end = min(gray.shape[0], y + patch_size)
            x_start = max(0, x - patch_size)
            x_end = min(gray.shape[1], x + patch_size)
            
            patch = gray[y_start:y_end, x_start:x_end]
            
            if patch.size > 0:
                # Apply Canny edge detection
                edges = cv2.Canny(patch, 50, 150)
                
                # Find closest edge point to estimated corner
                edge_points = np.column_stack(np.where(edges > 0))
                if len(edge_points) > 0:
                    center = np.array([y - y_start, x - x_start])
                    distances = np.linalg.norm(edge_points - center, axis=1)
                    closest_edge = edge_points[np.argmin(distances)]
                    
                    # Apply small correction (30% of the offset)
                    adjustment_y = (closest_edge[0] - center[0]) * 0.3
                    adjustment_x = (closest_edge[1] - center[1]) * 0.3
                    
                    refined_corner[0][j][1] += adjustment_y
                    refined_corner[0][j][0] += adjustment_x
        
        refined_corners_list.append(refined_corner)
    
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
    
    # Use contour-based refinement (better for perspective)
    aruco_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_CONTOUR
    aruco_params.cornerRefinementWinSize = 7
    aruco_params.cornerRefinementMaxIterations = 50
    aruco_params.cornerRefinementMinAccuracy = 0.005
    
    # Fine-tune detection sensitivity
    aruco_params.adaptiveThreshWinSizeMin = 3
    aruco_params.adaptiveThreshWinSizeMax = 25
    aruco_params.adaptiveThreshWinSizeStep = 4
    aruco_params.adaptiveThreshConstant = 5
    
    # Improve contour detection
    aruco_params.polygonalApproxAccuracyRate = 0.02
    aruco_params.minCornerDistanceRate = 0.03
    aruco_params.minDistanceToBorder = 1
    
    # Perimeter constraints
    aruco_params.minMarkerPerimeterRate = 0.02
    aruco_params.maxMarkerPerimeterRate = 4.0
    
    # Perspective handling
    aruco_params.perspectiveRemovePixelPerCell = 8
    aruco_params.perspectiveRemoveIgnoredMarginPerCell = 0.13
    
    # Error correction
    aruco_params.errorCorrectionRate = 0.8
    # ------------------------------------------
    
    corners, ids, rejected = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=aruco_params)
    
    pose_data = {}
    corners_data = {}
    
    if ids is not None:
        # --- STEP 1: ADVANCED SUBPIXEL + EDGE REFINEMENT ---
        print("Applying advanced corner refinement...")
        refined_corners_list = refine_corners_subpixel_advanced(gray, corners)
        
        # --- STEP 2: LINE INTERSECTION REFINEMENT ---
        print("Applying line intersection refinement...")
        final_corners_list = []
        for marker_corners_array in refined_corners_list:
            # Get the refined corners
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
                final_corners_list.append(new_corners)
            else:
                # Fallback to refined corners if line intersection fails
                final_corners_list.append(marker_corners_array)
        
        # Use the FINAL, highly accurate corners for all further steps
        corners = tuple(final_corners_list)
        print(f"Corner refinement complete. Detected {len(corners)} markers.")
        # ----------------------------------------------------

        ids_list = ids.flatten().tolist()
        id_counts = Counter(ids_list)

        if id_counts.get(5, 0) > 1 or id_counts.get(15, 0) > 1:
            print("Error: Duplicate marker IDs detected.")
            return None

        for i, marker_id_array in enumerate(ids):
            marker_id = marker_id_array[0]
            if marker_id == 5: 
                obj_points = objp_marker5
            elif marker_id == 15: 
                obj_points = objp_marker15
            else: 
                continue

            # Print corner coordinates for verification
            print(f"\nMarker {marker_id} refined corners:")
            for j, corner in enumerate(corners[i][0]):
                print(f"  Corner {j}: ({corner[0]:.3f}, {corner[1]:.3f})")

            success, rvec, tvec = cv2.solvePnP(obj_points, corners[i], camera_matrix, dist_coeffs)
            if success:
                pose_data[marker_id] = {'rvec': rvec, 'tvec': tvec}
                corners_data[marker_id] = corners[i]
    else:
        print("No markers detected.")
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
        
        print("\nPose estimation successful!")
        return results
    else:
        print("Error: Could not find both markers 5 and 15.")
        return None