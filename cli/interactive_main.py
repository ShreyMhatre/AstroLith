import argparse
import cv2
import sys
import os
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core import utils
from core import pose_estimator

def parse_args():
    parser = argparse.ArgumentParser(description="Interactively extend axes to measure box edges.")
    parser.add_argument('--profile', type=str, required=True, help="Path to the camera calibration profile file.")
    parser.add_argument('--image', type=str, required=True, help="Path to the input image file.")
    parser.add_argument('--marker-size', type=float, required=True, choices=[5, 10, 15, 20], help='Marker size in cm.')
    parser.add_argument('--aruco_dict', type=str, default='DICT_5X5_100', help='The ArUco dictionary to use.')
    return parser.parse_args()

def ray_plane_intersection(ray_origin, ray_direction, plane_origin, plane_normal):
    denom = np.dot(ray_direction, plane_normal)
    if np.abs(denom) < 1e-6: return None
    t = np.dot(plane_origin - ray_origin, plane_normal) / denom
    if t >= 0: return ray_origin + t * ray_direction
    return None

class InteractivePlot:
    def __init__(self, original_frame, results, marker_size_cm, camera_matrix, dist_coeffs):
        self.original_frame = original_frame
        self.results = results
        self.marker_size_cm = marker_size_cm
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
        self.fig, self.ax = plt.subplots(figsize=(12, 8))
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        self.reset_state()

    def reset_state(self):
        print("\n--- Resetting Axes (3D Mode) ---")
        self.origin_3d = self.results['common_origin']
        self.length_axis_3d = self.results['length_axis']
        self.width_axis_3d = self.results['width_axis']
        self.height_axis_3d = self.results['height_axis']
        marker_size_m = self.marker_size_cm / 100.0
        self.endpoints_3d = {
            'length': self.origin_3d + self.length_axis_3d * marker_size_m,
            'width': self.origin_3d + self.width_axis_3d * marker_size_m,
            'height': self.origin_3d + self.height_axis_3d * marker_size_m
        }
        self.edge_lengths_cm = {'Width': self.marker_size_cm, 'Length': self.marker_size_cm, 'Height': self.marker_size_cm}
        self.current_state = 'length'
        self.update_plot()
        
    def on_key_press(self, event):
        if event.key == 'r': self.reset_state()

    def on_click(self, event):
        if event.inaxes != self.ax or self.current_state == 'done': return
        click_point_2d = np.array([event.xdata, event.ydata], dtype=np.float32).reshape(1, 1, 2)
        undistorted_point_2d = cv2.undistortPoints(click_point_2d, self.camera_matrix, self.dist_coeffs, P=self.camera_matrix)[0][0]
        cam_pos_3d = np.array([0., 0., 0.])
        fx, fy = self.camera_matrix[0, 0], self.camera_matrix[1, 1]
        cx, cy = self.camera_matrix[0, 2], self.camera_matrix[1, 2]
        ray_direction = np.array([(undistorted_point_2d[0] - cx) / fx, (undistorted_point_2d[1] - cy) / fy, 1.0])
        ray_direction /= np.linalg.norm(ray_direction)
        floor_plane_normal = self.height_axis_3d
        vertical_plane_normal = np.cross(self.height_axis_3d, self.width_axis_3d)
        closest_axis = self.current_state
        intersection_3d = ray_plane_intersection(cam_pos_3d, ray_direction, self.origin_3d, floor_plane_normal if closest_axis != 'height' else vertical_plane_normal)
        if intersection_3d is not None:
            origin_to_intersect_vec = intersection_3d - self.origin_3d
            unit_axis_vec_map = {'length': self.length_axis_3d, 'width': self.width_axis_3d, 'height': self.height_axis_3d}
            unit_axis_vec = unit_axis_vec_map[closest_axis]
            projection_length_m = np.dot(origin_to_intersect_vec, unit_axis_vec)
            self.endpoints_3d[closest_axis] = self.origin_3d + unit_axis_vec * projection_length_m
            self.edge_lengths_cm[closest_axis.capitalize()] = projection_length_m * 100.0
            print(f"Updated {closest_axis.capitalize()}: {self.edge_lengths_cm[closest_axis.capitalize()]:.2f} cm")
        if self.current_state == 'length': self.current_state = 'width'
        elif self.current_state == 'width': self.current_state = 'height'
        elif self.current_state == 'height': self.current_state = 'done'
        self.update_plot()

    def update_plot(self):
        self.ax.clear()
        frame_with_outlines = self.original_frame.copy()
        rvec = np.zeros((3, 1), dtype=np.float32)
        tvec = np.zeros((3, 1), dtype=np.float32)
        origin_2d, _ = cv2.projectPoints(self.origin_3d.reshape(1,3), rvec, tvec, self.camera_matrix, self.dist_coeffs)
        endpoint_w_2d, _ = cv2.projectPoints(self.endpoints_3d['width'].reshape(1,3), rvec, tvec, self.camera_matrix, self.dist_coeffs)
        endpoint_l_2d, _ = cv2.projectPoints(self.endpoints_3d['length'].reshape(1,3), rvec, tvec, self.camera_matrix, self.dist_coeffs)
        endpoint_h_2d, _ = cv2.projectPoints(self.endpoints_3d['height'].reshape(1,3), rvec, tvec, self.camera_matrix, self.dist_coeffs)
        p_origin = tuple(origin_2d[0][0].astype(int))
        p_width = tuple(endpoint_w_2d[0][0].astype(int))
        p_length = tuple(endpoint_l_2d[0][0].astype(int))
        p_height = tuple(endpoint_h_2d[0][0].astype(int))
        cv2.line(frame_with_outlines, p_origin, p_width, (255, 0, 0), 3)
        cv2.line(frame_with_outlines, p_origin, p_length, (0, 0, 255), 3)
        cv2.line(frame_with_outlines, p_origin, p_height, (0, 255, 0), 3)
        cv2.circle(frame_with_outlines, p_origin, 8, (255, 255, 0), -1)
        y_pos = 0.05
        for name, length in self.edge_lengths_cm.items():
            self.ax.text(0.02, 1 - y_pos, f"{name}: {length:.2f} cm", color='cyan', fontsize=12, transform=self.ax.transAxes, bbox=dict(facecolor='black', alpha=0.5))
            y_pos += 0.05
        title = "Measurement Complete! | Press 'R' to reset" if self.current_state == 'done' else f"Click to measure the {self.current_state.upper()} edge"
        self.ax.set_title(title)
        self.ax.imshow(cv2.cvtColor(frame_with_outlines, cv2.COLOR_BGR2RGB))
        self.ax.axis('off')
        self.fig.canvas.draw()

def main():
    args = parse_args()
    camera_matrix, dist_coeffs = utils.load_calibration_profile(args.profile)
    if camera_matrix is None: sys.exit(1)
    aruco_dict = utils.get_aruco_dictionary(args.aruco_dict)
    if aruco_dict is None: sys.exit(1)
    original_frame = cv2.imread(args.image)
    if original_frame is None: return
    marker_size_cm = args.marker_size
    marker_size_meters = marker_size_cm / 100.0
    results = pose_estimator.find_box_pose(original_frame, camera_matrix, dist_coeffs, aruco_dict, marker_size_meters)
    if not results:
        print("Could not find the required markers in the image. Exiting.")
        return
    plot = InteractivePlot(original_frame, results, marker_size_cm, camera_matrix, dist_coeffs)
    plt.show()
    print("Program finished.")

if __name__ == '__main__':
    main()