import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
image = cv2.imread('assets/demo/box1.jpeg')
original = image.copy()

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply bilateral filter to preserve edges while reducing noise
gray_filtered = cv2.bilateralFilter(gray, 9, 75, 75)

# Load dictionary
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)

# Configure detection parameters for better edge detection
parameters = cv2.aruco.DetectorParameters()

# More aggressive corner refinement
parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_CONTOUR
parameters.cornerRefinementWinSize = 7
parameters.cornerRefinementMaxIterations = 50
parameters.cornerRefinementMinAccuracy = 0.005

# Fine-tune detection sensitivity
parameters.adaptiveThreshWinSizeMin = 3
parameters.adaptiveThreshWinSizeMax = 25
parameters.adaptiveThreshWinSizeStep = 4
parameters.adaptiveThreshConstant = 5

# Improve contour detection
parameters.polygonalApproxAccuracyRate = 0.02
parameters.minCornerDistanceRate = 0.03
parameters.minDistanceToBorder = 1

# Perimeter constraints
parameters.minMarkerPerimeterRate = 0.02
parameters.maxMarkerPerimeterRate = 4.0

# Perspective handling
parameters.perspectiveRemovePixelPerCell = 8
parameters.perspectiveRemoveIgnoredMarginPerCell = 0.13

# Error correction
parameters.errorCorrectionRate = 0.8

# Create detector
detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

# Detect markers
corners, ids, rejected = detector.detectMarkers(gray_filtered)

print(f"Detected {len(ids) if ids is not None else 0} markers")
if ids is not None:
    print("Marker IDs:", ids.flatten())
    
    # Apply multiple rounds of subpixel refinement
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.00001)
    
    refined_corners = []
    for i, corner in enumerate(corners):
        # First pass: standard subpixel
        refined1 = cv2.cornerSubPix(
            gray_filtered,
            corner.copy(),
            winSize=(7, 7),
            zeroZone=(-1, -1),
            criteria=criteria
        )
        
        # Second pass: on original gray image for comparison
        refined2 = cv2.cornerSubPix(
            gray,
            refined1.copy(),
            winSize=(5, 5),
            zeroZone=(-1, -1),
            criteria=criteria
        )
        
        # Use edge detection to further refine
        # Create a local patch around each corner
        refined_corner = refined2.copy()
        for j, point in enumerate(refined2[0]):
            x, y = int(point[0]), int(point[1])
            
            # Extract local patch (20x20 pixels around corner)
            patch_size = 20
            y_start = max(0, y - patch_size)
            y_end = min(gray.shape[0], y + patch_size)
            x_start = max(0, x - patch_size)
            x_end = min(gray.shape[1], x + patch_size)
            
            patch = gray[y_start:y_end, x_start:x_end]
            
            if patch.size > 0:
                # Apply Canny edge detection to find actual edges
                edges = cv2.Canny(patch, 50, 150)
                
                # Find the strongest edge near the estimated corner
                edge_points = np.column_stack(np.where(edges > 0))
                if len(edge_points) > 0:
                    # Find closest edge point to center
                    center = np.array([patch_size, patch_size])
                    if y_start == 0:
                        center[0] = y
                    if x_start == 0:
                        center[1] = x
                    
                    distances = np.linalg.norm(edge_points - center, axis=1)
                    closest_edge = edge_points[np.argmin(distances)]
                    
                    # Adjust corner position (small correction only)
                    adjustment_y = (closest_edge[0] - (y - y_start)) * 0.3
                    adjustment_x = (closest_edge[1] - (x - x_start)) * 0.3
                    
                    refined_corner[0][j][0] += adjustment_x
                    refined_corner[0][j][1] += adjustment_y
        
        refined_corners.append(refined_corner)
        
        print(f"\nMarker {ids[i][0]} corners (final refined):")
        for j, (x, y) in enumerate(refined_corner[0]):
            print(f"  Corner {j}: ({x:.3f}, {y:.3f})")
    
    corners_final = np.array(refined_corners)
else:
    corners_final = corners

# Visualize comparison
fig, axes = plt.subplots(1, 3, figsize=(20, 7))

# Original detection
img1 = original.copy()
if ids is not None:
    cv2.aruco.drawDetectedMarkers(img1, corners, ids)
axes[0].imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
axes[0].set_title(f'Initial Detection\n{len(ids) if ids is not None else 0} markers')
axes[0].axis('off')

# Final refined detection with corner points
img2 = original.copy()
if ids is not None:
    cv2.aruco.drawDetectedMarkers(img2, corners_final, ids)
    # Draw refined corner points
    for corner in corners_final:
        for point in corner[0]:
            cv2.circle(img2, tuple(point.astype(int)), 4, (0, 255, 0), -1)
            cv2.circle(img2, tuple(point.astype(int)), 6, (255, 255, 0), 2)
axes[1].imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
axes[1].set_title(f'Edge-Refined Detection\n{len(ids) if ids is not None else 0} markers')
axes[1].axis('off')

# Zoomed view of first marker
if ids is not None and len(ids) > 0:
    # Find bounding box of first marker
    pts = corners_final[0][0].astype(int)
    x_min, y_min = pts.min(axis=0) - 50
    x_max, y_max = pts.max(axis=0) + 50
    
    # Ensure bounds are valid
    x_min = max(0, x_min)
    y_min = max(0, y_min)
    x_max = min(img2.shape[1], x_max)
    y_max = min(img2.shape[0], y_max)
    
    zoomed = img2[y_min:y_max, x_min:x_max].copy()
    axes[2].imshow(cv2.cvtColor(zoomed, cv2.COLOR_BGR2RGB))
    axes[2].set_title(f'Zoomed: Marker {ids[0][0]}')
else:
    axes[2].imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    axes[2].set_title('No markers detected')
axes[2].axis('off')

plt.tight_layout()
plt.show()

# Measurement analysis
if ids is not None and len(ids) > 0:
    print("\n=== Measurement Analysis ===")
    
    # Function to expand corners to include white border
    def expand_to_outer_edge(corners, expansion_factor=1.2):
        """
        Expands detected corners outward to capture the full marker including white border.
        expansion_factor: typically 1.15-1.25 depending on border thickness
        """
        expanded = []
        for corner in corners:
            pts = corner[0]
            # Calculate center of marker
            center = np.mean(pts, axis=0)
            
            # Expand each corner away from center
            expanded_pts = []
            for pt in pts:
                direction = pt - center
                expanded_pt = center + direction * expansion_factor
                expanded_pts.append(expanded_pt)
            
            expanded.append(np.array([expanded_pts], dtype=np.float32))
        
        return np.array(expanded)
    
    # Create expanded corners for outer edge detection
    corners_outer = expand_to_outer_edge(corners_final, expansion_factor=1.2)
    
    for i in range(len(ids)):
        print(f"\nMarker {ids[i][0]}:")
        
        # Inner (detected) corners
        pts_inner = corners_final[i][0]
        side1_inner = np.linalg.norm(pts_inner[0] - pts_inner[1])
        side2_inner = np.linalg.norm(pts_inner[1] - pts_inner[2])
        side3_inner = np.linalg.norm(pts_inner[2] - pts_inner[3])
        side4_inner = np.linalg.norm(pts_inner[3] - pts_inner[0])
        avg_inner = np.mean([side1_inner, side2_inner, side3_inner, side4_inner])
        
        print(f"  Inner (black square) avg: {avg_inner:.2f} pixels")
        
        # Outer (expanded) corners
        pts_outer = corners_outer[i][0]
        side1_outer = np.linalg.norm(pts_outer[0] - pts_outer[1])
        side2_outer = np.linalg.norm(pts_outer[1] - pts_outer[2])
        side3_outer = np.linalg.norm(pts_outer[2] - pts_outer[3])
        side4_outer = np.linalg.norm(pts_outer[3] - pts_outer[0])
        avg_outer = np.mean([side1_outer, side2_outer, side3_outer, side4_outer])
        
        print(f"  Outer (with border) avg: {avg_outer:.2f} pixels")
        print(f"  Border width: {(avg_outer - avg_inner) / 2:.2f} pixels per side")
        
        # Check squareness
        std_dev = np.std([side1_inner, side2_inner, side3_inner, side4_inner])
        print(f"  Side length variation (std): {std_dev:.2f} pixels")
        
        # Check angles (should be ~90 degrees for a square)
        def angle_between_vectors(v1, v2):
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            return np.degrees(np.arccos(np.clip(cos_angle, -1, 1)))
        
        v1 = pts_inner[1] - pts_inner[0]
        v2 = pts_inner[3] - pts_inner[0]
        angle1 = angle_between_vectors(v1, v2)
        
        v3 = pts_inner[2] - pts_inner[1]
        v4 = pts_inner[0] - pts_inner[1]
        angle2 = angle_between_vectors(v3, v4)
        
        print(f"  Corner angles: {angle1:.1f}°, {angle2:.1f}° (should be ~90°)")
        print(f"  Angle error: {abs(90-angle1):.1f}°, {abs(90-angle2):.1f}°")
    
    # Visualize outer edges
    print("\n=== Visualizing Outer Edge Detection ===")
    img_comparison = original.copy()
    
    if ids is not None:
        for i in range(len(ids)):
            # Draw inner corners (green)
            pts_inner = corners_final[i][0].astype(int)
            cv2.polylines(img_comparison, [pts_inner], True, (0, 255, 0), 2)
            
            # Draw outer corners (cyan)
            pts_outer = corners_outer[i][0].astype(int)
            cv2.polylines(img_comparison, [pts_outer], True, (255, 255, 0), 2)
            
            # Draw corner points
            for pt in pts_inner:
                cv2.circle(img_comparison, tuple(pt), 4, (0, 255, 0), -1)
            for pt in pts_outer:
                cv2.circle(img_comparison, tuple(pt), 4, (255, 255, 0), -1)
    
    plt.figure(figsize=(12, 8))
    plt.imshow(cv2.cvtColor(img_comparison, cv2.COLOR_BGR2RGB))
    plt.title('Green = Inner (detected), Cyan = Outer (estimated with border)')
    plt.axis('off')
    plt.tight_layout()
    plt.show()