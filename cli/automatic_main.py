import cv2
import numpy as np
import argparse
import sys
import os
import matplotlib.pyplot as plt

# Add the parent directory to the path to find the 'core' module
# Make sure the 'core' module is in the parent directory of your script's location
# For example, if your script is in /my_project/scripts/, 'core' should be in /my_project/
try:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from core import utils
    from core import pose_estimator
except ImportError:
    print("Error: Could not import the 'core' module.")
    print("Please ensure your script is in a subdirectory and the 'core' module is in the parent directory.")
    sys.exit(1)


# --- Tunable Parameters for Initial Automatic Detection ---
ROI_WIDTH = 40
CANNY_LOW_THRESHOLD = 50
CANNY_HIGH_THRESHOLD = 150
HOUGH_THRESHOLD = 20
HOUGH_MIN_LINE_LENGTH = 30
HOUGH_MAX_LINE_GAP = 10
PARALLELISM_THRESHOLD = 0.98
# Volumetric weight divisor (e.g., 5000 is common for shipping)
VOLUMETRIC_DIVISOR = 5000


def dist_point_to_segment(p, a, b):
    """Calculates the minimum distance from a point p to a line segment ab."""
    if np.all(a == b): return np.linalg.norm(p - a)
    # Normalize the direction vector of the segment
    segment_vec = b - a
    norm = np.linalg.norm(segment_vec)
    if norm == 0: return np.linalg.norm(p - a)
    d = segment_vec / norm
    # Project p onto the line defined by a and d
    t = np.dot(p - a, d)
    # If the projection is before a, the closest point is a
    if t < 0: return np.linalg.norm(p - a)
    # If the projection is after b, the closest point is b
    if t > norm: return np.linalg.norm(p - b)
    # Otherwise, the closest point is the projection itself
    return np.linalg.norm(p - (a + t * d))


class HybridMeasurement:
    def __init__(self, image, results, camera_matrix, dist_coeffs):
        self.image = image
        self.results = results
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
        self.measurements_cm = {}
        self.endpoints_3d = {}
        self.endpoints_2d = {}
        self.origin_2d = None
        self.run_automatic_measurement()
        self.fig, self.ax = plt.subplots(figsize=(12, 9))
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.update_plot()
        plt.show()

    @staticmethod
    def ray_plane_intersection(ray_origin, ray_direction, plane_origin, plane_normal):
        """Calculates the 3D intersection of a ray and a plane."""
        denom = np.dot(ray_direction, plane_normal)
        if np.abs(denom) < 1e-6: return None
        t = np.dot(plane_origin - ray_origin, plane_normal) / denom
        if t >= 0: return ray_origin + t * ray_direction
        return None

    def run_automatic_measurement(self):
        print("--- Running Initial Automatic Measurement ---")
        origin_3d = self.results['common_origin']
        rvec = np.zeros((3, 1)); tvec = np.zeros((3, 1))
        origin_2d_tuple, _ = cv2.projectPoints(origin_3d.reshape(1, 3), rvec, tvec, self.camera_matrix, self.dist_coeffs)
        self.origin_2d = origin_2d_tuple[0][0]

        axis_vectors_2d = {}
        for axis in ['width', 'length', 'height']:
            endpoint_3d = origin_3d + self.results[f'{axis}_axis'] * 0.1
            endpoint_2d_tuple, _ = cv2.projectPoints(endpoint_3d.reshape(1, 3), rvec, tvec, self.camera_matrix, self.dist_coeffs)
            vec_2d = endpoint_2d_tuple[0][0] - self.origin_2d
            axis_vectors_2d[axis] = vec_2d / np.linalg.norm(vec_2d)

        for axis in ['width', 'length', 'height']:
            print(f"Finding '{axis}' edge...")
            endpoint_2d = self.find_edge_endpoint_automatically(axis_vectors_2d[axis])
            if endpoint_2d is not None:
                length_cm, endpoint_3d = self.get_3d_measurement(endpoint_2d, axis)
                if length_cm is not None:
                    self.measurements_cm[axis.capitalize()] = length_cm
                    self.endpoints_3d[axis] = endpoint_3d
            else:
                print(f" -> Could not automatically find '{axis}' edge. Setting to default.")
                self.endpoints_3d[axis] = origin_3d + self.results[f'{axis}_axis'] * 0.05
                self.measurements_cm[axis.capitalize()] = 5.0
        self.project_all_endpoints()

    def on_click(self, event):
        if event.inaxes != self.ax: return
        click_point = np.array([event.xdata, event.ydata])
        distances_to_axes = {axis: dist_point_to_segment(click_point, self.origin_2d, endpoint) for axis, endpoint in self.endpoints_2d.items()}
        closest_axis = min(distances_to_axes, key=distances_to_axes.get)
        print(f"\nUser correction for '{closest_axis}' axis.")
        length_cm, endpoint_3d = self.get_3d_measurement(click_point, closest_axis)
        if length_cm is not None:
            self.measurements_cm[closest_axis.capitalize()] = length_cm
            self.endpoints_3d[closest_axis] = endpoint_3d
            self.project_all_endpoints()
            self.update_plot()

    def update_plot(self):
        self.ax.clear()
        self.ax.imshow(cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB))

        # --- Parameters for Axes Representation (Editable) ---
        axes_params = {
            'width': {'color': 'blue', 'linewidth': 2},
            'length': {'color': 'red', 'linewidth': 2},
            'height': {'color': 'lime', 'linewidth': 2}
        }
        origin_marker_size = 25
        origin_marker_color = 'cyan'
        origin_marker_edge_color = 'black'

        # --- Parameters for Text Display (Editable) ---
        text_font_size = 8
        text_color = 'cyan'
        text_start_x_pos = 0.02  # Percentage from left
        text_start_y_pos = 0.95  # Percentage from top
        text_y_step = 0.05      # Step down for each new line
        text_bbox_props = dict(facecolor='black', alpha=0.7)

        # Draw measurement axes
        for axis, endpoint_2d in self.endpoints_2d.items():
            params = axes_params.get(axis, {})
            self.ax.plot([self.origin_2d[0], endpoint_2d[0]],
                         [self.origin_2d[1], endpoint_2d[1]],
                         color=params.get('color', 'white'),
                         linewidth=params.get('linewidth', 1))

        # Draw the common origin point
        self.ax.scatter(self.origin_2d[0], self.origin_2d[1],
                        c=origin_marker_color, s=origin_marker_size,
                        zorder=5, edgecolors=origin_marker_edge_color)

        # --- Text and Measurement Display ---
        display_texts = []
        # Define the desired order
        measurement_order = ['Width', 'Length', 'Height']
        for name in measurement_order:
            if name in self.measurements_cm:
                length = self.measurements_cm[name]
                display_texts.append(f"{name}: {length:.2f} cm")

        # Calculate and add Volumetric Weight
        if all(k in self.measurements_cm for k in ('Width', 'Length', 'Height')):
            w = self.measurements_cm['Width']
            l = self.measurements_cm['Length']
            h = self.measurements_cm['Height']
            volumetric_weight = (w * l * h) / VOLUMETRIC_DIVISOR
            display_texts.append(f"Volumetric Weight: {volumetric_weight:.2f} kg")

        # Display all text items
        y_pos = text_start_y_pos
        for text in display_texts:
            self.ax.text(text_start_x_pos, y_pos, text, color=text_color,
                         fontsize=text_font_size, transform=self.ax.transAxes,
                         bbox=text_bbox_props)
            y_pos -= text_y_step

        self.ax.set_title("Automatic Measurement | Click any axis to correct")
        self.ax.axis('off')
        self.fig.canvas.draw()

    def project_all_endpoints(self):
        rvec = np.zeros((3, 1)); tvec = np.zeros((3, 1))
        for axis, endpoint_3d in self.endpoints_3d.items():
            ep_2d, _ = cv2.projectPoints(endpoint_3d.reshape(1, 3), rvec, tvec, self.camera_matrix, self.dist_coeffs)
            self.endpoints_2d[axis] = ep_2d[0][0]

    def get_3d_measurement(self, endpoint_2d, axis):
        origin_3d = self.results['common_origin']
        axis_3d = self.results[f'{axis}_axis']
        # Define the plane for ray intersection
        plane_normal = self.results['height_axis'] if axis != 'height' else np.cross(self.results['height_axis'], self.results['width_axis'])
        # Get the ray from the camera through the clicked 2D point
        undistorted_point = cv2.undistortPoints(np.array([[endpoint_2d]], dtype=np.float32), self.camera_matrix, self.dist_coeffs, P=self.camera_matrix)[0][0]
        cam_pos_3d = np.array([0., 0., 0.])
        fx, fy, cx, cy = self.camera_matrix[0, 0], self.camera_matrix[1, 1], self.camera_matrix[0, 2], self.camera_matrix[1, 2]
        ray_dir = np.array([(undistorted_point[0] - cx) / fx, (undistorted_point[1] - cy) / fy, 1.0])
        ray_dir /= np.linalg.norm(ray_dir)
        # Find where the ray intersects the plane
        intersection_3d = self.ray_plane_intersection(cam_pos_3d, ray_dir, origin_3d, plane_normal)
        if intersection_3d is None: return None, None
        # Project the intersection point onto the measurement axis
        origin_to_intersect_vec = intersection_3d - origin_3d
        projection_length_m = np.dot(origin_to_intersect_vec, axis_3d)
        final_endpoint_3d = origin_3d + axis_3d * projection_length_m
        return projection_length_m * 100.0, final_endpoint_3d

    def find_edge_endpoint_automatically(self, axis_unit_vec_2d):
        # Create a narrow rectangular ROI along the axis
        mask = np.zeros(self.image.shape[:2], dtype=np.uint8)
        far_point = self.origin_2d + axis_unit_vec_2d * 2000  # A point very far away
        perp_vec = np.array([-axis_unit_vec_2d[1], axis_unit_vec_2d[0]]) # Perpendicular vector
        roi_corners = np.array([[self.origin_2d - perp_vec * (ROI_WIDTH / 2),
                                 far_point - perp_vec * (ROI_WIDTH / 2),
                                 far_point + perp_vec * (ROI_WIDTH / 2),
                                 self.origin_2d + perp_vec * (ROI_WIDTH / 2)]], dtype=np.int32)
        cv2.fillPoly(mask, roi_corners, 255)

        # Detect lines within the ROI
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, CANNY_LOW_THRESHOLD, CANNY_HIGH_THRESHOLD)
        masked_edges = cv2.bitwise_and(edges, edges, mask=mask)
        lines = cv2.HoughLinesP(masked_edges, 1, np.pi / 180, HOUGH_THRESHOLD, HOUGH_MIN_LINE_LENGTH, HOUGH_MAX_LINE_GAP)

        if lines is None: return None

        # Find the best line that is parallel to the axis
        best_line = None; max_score = 0
        for line in lines:
            x1, y1, x2, y2 = line[0]
            line_vec = np.array([x2 - x1, y2 - y1])
            line_length = np.linalg.norm(line_vec)
            if line_length == 0: continue
            line_unit_vec = line_vec / line_length
            parallelism = abs(np.dot(line_unit_vec, axis_unit_vec_2d))
            # Score considers both line length and how parallel it is to the target axis
            score = (line_length * 0.7) + (parallelism * 0.3)
            if parallelism > PARALLELISM_THRESHOLD and score > max_score:
                max_score = score
                best_line = line[0]

        if best_line is None: return None

        # Find the farthest point on the detected edge line from the origin
        x1, y1, x2, y2 = best_line
        line_p1 = np.array([x1, y1])
        line_dir = np.array([x2 - x1, y2 - y1])
        norm = np.linalg.norm(line_dir)
        if norm == 0: return None
        
        # ============================ FIX: Use standard division to create a new float array ============================
        line_dir = line_dir / norm # Normalize
        # =================================================================================================================

        # Project all edge points onto the best line to find the endpoint
        edge_points = np.argwhere(masked_edges > 0)[:, ::-1] # (x, y) format
        if len(edge_points) == 0: return None
        vectors_from_p1 = edge_points - line_p1
        projections = line_p1 + np.dot(vectors_from_p1, line_dir)[:, np.newaxis] * line_dir
        distances = np.linalg.norm(projections - self.origin_2d, axis=1)
        farthest_point = projections[np.argmax(distances)]

        return farthest_point


def main():
    parser = argparse.ArgumentParser(description="Automatically measure box edges with a manual correction option.")
    parser.add_argument('--profile', type=str, required=True, help="Path to the camera calibration profile file.")
    parser.add_argument('--image', type=str, required=True, help="Path to the input image file.")
    parser.add_argument('--marker-size', type=float, required=True, help='Marker size in cm.')
    parser.add_argument('--aruco_dict', type=str, default='DICT_5X5_100', help='The ArUco dictionary to use.')
    args = parser.parse_args()

    # Load camera profile and image
    camera_matrix, dist_coeffs = utils.load_calibration_profile(args.profile)
    aruco_dict = utils.get_aruco_dictionary(args.aruco_dict)
    image = cv2.imread(args.image)
    if image is None:
        print(f"Error: Could not read image at {args.image}")
        return

    # Find the pose of the box from ArUco markers
    results = pose_estimator.find_box_pose(image, camera_matrix, dist_coeffs, aruco_dict, args.marker_size / 100.0)
    if not results:
        print("Could not find markers for pose estimation.")
        return

    # Start the measurement process
    HybridMeasurement(image, results, camera_matrix, dist_coeffs)


if __name__ == '__main__':
    main()