# In cli/interactive_main.py

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
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Interactively extend axes to measure box edges.")
    parser.add_argument('--profile', type=str, required=True, help="Path to the camera calibration profile file.")
    parser.add_argument('--image', type=str, required=True, help="Path to the input image file.")
    parser.add_argument('--marker-size', type=float, required=True, choices=[5, 10, 15, 20], help='Marker size in cm.')
    parser.add_argument('--aruco_dict', type=str, default='DICT_5X5_100', help='The ArUco dictionary to use.')
    return parser.parse_args()

def dist_point_to_segment(p, a, b):
    """Calculates the minimum distance from a point p to a line segment ab."""
    if np.all(a == b): return np.linalg.norm(p - a)
    # Normalized tangent vector
    d = (b - a) / np.linalg.norm(b - a)
    # Signed parallel distance
    t = (p - a).dot(d)
    # If the projection is outside the segment, clamp to the nearest endpoint
    if t < 0: return np.linalg.norm(p - a)
    if t > np.linalg.norm(b - a): return np.linalg.norm(p - b)
    # The projection is inside the segment
    return np.linalg.norm(p - (a + t * d))

class InteractivePlot:
    def __init__(self, original_frame, results, marker_size_cm):
        self.original_frame = original_frame
        self.results = results
        self.marker_size_cm = marker_size_cm
        
        # --- CORE FIX: Reorder the initialization ---
        # 1. First, create the figure and axes so they exist.
        self.fig, self.ax = plt.subplots(figsize=(12, 8))
        
        # 2. Then, connect the event handlers.
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        
        # 3. Now that the plot exists, initialize the application state.
        #    This will call update_plot(), which can now safely access self.ax.
        self.reset_state()
        # ----------------------------------------------

    def reset_state(self):
        """Resets the measurement state to the beginning."""
        print("\n--- Resetting Axes ---")
        # Initialize 2D vectors and lengths from high-accuracy corners
        corners5 = self.results['corners_data'][5][0].astype(int)
        corners15 = self.results['corners_data'][15][0].astype(int)
        self.common_origin_2d = ((corners5[2] + corners15[3]) / 2.0).astype(int)
        
        width_dir_vec = corners5[3] - corners5[2]
        length_dir_vec = corners15[2] - corners15[3]
        height_dir_vec_5 = corners5[1] - corners5[2]
        height_dir_vec_15 = corners15[0] - corners15[3]
        avg_height_dir_vec = ((height_dir_vec_5 + height_dir_vec_15) / 2.0).astype(int)
        
        self.dir_vectors = {
            'width': width_dir_vec, 'length': length_dir_vec, 'height': avg_height_dir_vec
        }
        self.unit_dir_vectors = {k: v / np.linalg.norm(v) for k, v in self.dir_vectors.items()}
        
        self.initial_pixel_lengths = {
            'width': np.linalg.norm(width_dir_vec),
            'length': np.linalg.norm(length_dir_vec),
            'height': np.linalg.norm(avg_height_dir_vec)
        }

        self.edge_endpoints = {
            'width': self.common_origin_2d + width_dir_vec,
            'length': self.common_origin_2d + length_dir_vec,
            'height': self.common_origin_2d + avg_height_dir_vec
        }
        self.edge_lengths_cm = {
            'Width': self.marker_size_cm, 'Length': self.marker_size_cm, 'Height': self.marker_size_cm
        }
        
        self.current_state = 'length' # Start by measuring length
        self.update_plot()
        
    def on_key_press(self, event):
        """Handle key presses to reset the state."""
        if event.key == 'r':
            self.reset_state()

    def on_click(self, event):
        """Handle mouse clicks to extend axes."""
        if event.inaxes != self.ax: return
        if self.current_state == 'done': return

        click_point = np.array([event.xdata, event.ydata])

        # Find which axis the user is trying to extend
        distances_to_axes = {
            'length': dist_point_to_segment(click_point, self.common_origin_2d, self.edge_endpoints['length']),
            'width': dist_point_to_segment(click_point, self.common_origin_2d, self.edge_endpoints['width']),
            'height': dist_point_to_segment(click_point, self.common_origin_2d, self.edge_endpoints['height']),
        }
        closest_axis = min(distances_to_axes, key=distances_to_axes.get)

        # Vector Projection to Extend the Line
        unit_axis_vec = self.unit_dir_vectors[closest_axis]
        origin_to_click_vec = click_point - self.common_origin_2d
        projection_length = origin_to_click_vec.dot(unit_axis_vec)
        
        new_endpoint = self.common_origin_2d + unit_axis_vec * projection_length
        self.edge_endpoints[closest_axis] = new_endpoint.astype(int)

        # Calculate and update the real-world length
        new_pixel_length = np.linalg.norm(new_endpoint - self.common_origin_2d)
        initial_pixel_length = self.initial_pixel_lengths[closest_axis]
        new_cm_length = (new_pixel_length / initial_pixel_length) * self.marker_size_cm
        self.edge_lengths_cm[closest_axis.capitalize()] = new_cm_length
        
        print(f"Updated {closest_axis.capitalize()}: {new_cm_length:.2f} cm")
        
        # Advance the state
        if self.current_state == 'length': self.current_state = 'width'
        elif self.current_state == 'width': self.current_state = 'height'
        elif self.current_state == 'height': self.current_state = 'done'

        self.update_plot()

    def update_plot(self):
        """Redraws the entire plot based on the current state."""
        self.ax.clear()
        
        # Draw the base image with outlines
        frame_with_outlines = self.original_frame.copy()
        cv2.aruco.drawDetectedMarkers(frame_with_outlines, [self.results['corners_data'][5]], None)
        cv2.aruco.drawDetectedMarkers(frame_with_outlines, [self.results['corners_data'][15]], None)

        # Draw the extended axes
        cv2.line(frame_with_outlines, tuple(self.common_origin_2d), tuple(self.edge_endpoints['width']), (255, 0, 0), 3)
        cv2.line(frame_with_outlines, tuple(self.common_origin_2d), tuple(self.edge_endpoints['length']), (0, 0, 255), 3)
        cv2.line(frame_with_outlines, tuple(self.common_origin_2d), tuple(self.edge_endpoints['height']), (0, 255, 0), 3)
        cv2.circle(frame_with_outlines, tuple(self.common_origin_2d), 8, (255, 255, 0), -1)

        # Draw the length text
        y_pos = 0.05
        for name, length in self.edge_lengths_cm.items():
            text = f"{name}: {length:.2f} cm"
            self.ax.text(0.02, 1 - y_pos, text, color='cyan', fontsize=12, transform=self.ax.transAxes,
                         bbox=dict(facecolor='black', alpha=0.5))
            y_pos += 0.05
            
        # Set the title to guide the user
        if self.current_state == 'done':
            title = "Measurement Complete! | Press 'R' to reset | Close window to quit"
        else:
            title = f"Click near the {self.current_state.upper()} axis to extend it"
        self.ax.set_title(title)
        
        # Display the final image
        rgb_frame = cv2.cvtColor(frame_with_outlines, cv2.COLOR_BGR2RGB)
        self.ax.imshow(rgb_frame)
        self.ax.axis('off')
        self.fig.canvas.draw()

def main():
    args = parse_args()
    
    # --- 1. Load Data and Perform Initial Pose Estimation ---
    camera_matrix, dist_coeffs = utils.load_calibration_profile(args.profile)
    if camera_matrix is None: sys.exit(1)
    
    aruco_dict = utils.get_aruco_dictionary(args.aruco_dict)
    if aruco_dict is None: sys.exit(1)
    
    original_frame = cv2.imread(args.image)
    if original_frame is None: return
        
    marker_size_cm = args.marker_size
    marker_size_meters = marker_size_cm / 100.0

    results = pose_estimator.find_box_pose(
        original_frame, camera_matrix, dist_coeffs, aruco_dict, marker_size_meters
    )
    if not results:
        print("Could not find the required markers in the image. Exiting.")
        return

    # --- 2. Launch the Interactive Plot ---
    plot = InteractivePlot(original_frame, results, marker_size_cm)
    plt.show()

    print("Program finished.")

if __name__ == '__main__':
    main()