import cv2
import numpy as np
import argparse
import time
import os
import matplotlib.pyplot as plt

def load_camera_parameters(profile_path):
    """
    Loads camera calibration parameters from a file.
    Supports .npz, .yml, and .xml formats.

    Args:
        profile_path (str): Path to the calibration file.

    Returns:
        tuple: A tuple containing the camera matrix and distortion coefficients.
               Returns (None, None) if loading fails.
    """
    if not os.path.exists(profile_path):
        print(f"Error: Profile file not found at {profile_path}")
        return None, None
        
    if profile_path.endswith('.npz'):
        try:
            data = np.load(profile_path)
            camera_matrix = data['mtx']
            dist_coeffs = data['dist']
            print("Loaded camera profile from .npz file.")
            return camera_matrix, dist_coeffs
        except Exception as e:
            print(f"Error loading .npz file: {e}")
            print("Ensure the .npz file contains 'mtx' and 'dist' arrays.")
            return None, None
            
    elif profile_path.endswith(('.yml', '.xml')):
        try:
            fs = cv2.FileStorage(profile_path, cv2.FILE_STORAGE_READ)
            camera_matrix = fs.getNode("camera_matrix").mat()
            dist_coeffs = fs.getNode("distortion_coefficients").mat()
            fs.release()
            print("Loaded camera profile from .yml/.xml file.")
            return camera_matrix, dist_coeffs
        except Exception as e:
            print(f"Error loading YAML/XML file: {e}")
            return None, None
    else:
        print(f"Error: Unsupported profile file format: {os.path.basename(profile_path)}")
        return None, None

def aruco_pose_estimation(image_path, profile_path=None, marker_size=0.15):
    """
    Performs pose estimation on ArUco markers following a detailed step-by-step process.

    Args:
        image_path (str): Path to the input image.
        profile_path (str, optional): Path to the camera calibration file. Defaults to None.
        marker_size (float, optional): The real-world size of the marker side in meters. Defaults to 0.15.
    """
    print("Starting ArUco detection process...")
    start_total_time = time.perf_counter()

    # --- Load Image ---
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load image from path: {image_path}")
        return

    # 1. Convert to Grey
    print("(detect|ConvertGrey)")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 2-7. Marker Detection
    print("(detect|Threshold and Detect rectangles ... Corner Refinement)")
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_ARUCO_ORIGINAL)
    parameters = cv2.aruco.DetectorParameters()
    parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX

    detection_start_time = time.perf_counter()
    corners, ids, rejected = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
    detection_end_time = time.perf_counter()
    
    detection_duration_ms = (detection_end_time - detection_start_time) * 1000
    print(f"Time detection={detection_duration_ms:.4f} milliseconds nmarkers={len(ids) if ids is not None else 0}")

    # 8. Pose Estimation
    if ids is not None:
        cv2.aruco.drawDetectedMarkers(img, corners, ids) # Draw markers regardless of profile
        if profile_path:
            print("(detect|Pose Estimation)")
            camera_matrix, dist_coeffs = load_camera_parameters(profile_path)

            if camera_matrix is not None and dist_coeffs is not None:
                rvecs, tvecs, _objPoints = cv2.aruco.estimatePoseSingleMarkers(
                    corners, marker_size, camera_matrix, dist_coeffs
                )

                for i, marker_id in enumerate(ids):
                    tvec, rvec = tvecs[i][0], rvecs[i][0]
                    # Draw the 3D axes on the image
                    cv2.drawFrameAxes(img, camera_matrix, dist_coeffs, rvec, tvec, marker_size / 2)
                    print(f"{marker_id[0]}=({corners[i][0][0][0]:.3f},{corners[i][0][0][1]:.3f}) ... Txyz={tvec[0]:.6f} {tvec[1]:.6f} {tvec[2]:.6f} Rxyz={rvec[0]:.6f} {rvec[1]:.6f} {rvec[2]:.6f}")
            else:
                print("Could not process camera profile. Displaying detected markers only.")
        else:
            print("No camera profile provided. Displaying detected markers only.")
    else:
        print("No markers found.")

    # 9. Total
    end_total_time = time.perf_counter()
    total_duration_ms = (end_total_time - start_total_time) * 1000
    print(f"(detect|total): {total_duration_ms:.4f}ms")

    # --- Display the final image using Matplotlib ---
    print("\nDisplaying output image...")
    # Convert from BGR (OpenCV default) to RGB (Matplotlib default)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    plt.figure(figsize=(10, 8)) # Optional: Adjust figure size
    plt.imshow(img_rgb)
    plt.title("ArUco Pose Estimation Output")
    plt.axis('off') # Hide the axes
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Perform ArUco marker pose estimation using an image and camera profile."
    )
    parser.add_argument("--image", type=str, required=True, help="Path to the input image file.")
    parser.add_argument("--profile", type=str, help="Path to camera profile (.npz, .yml, or .xml).")
    parser.add_argument("-s", "--size", type=float, default=0.15, help="Real-world marker size in meters. Default: 0.15")
    args = parser.parse_args()
    
    aruco_pose_estimation(args.image, args.profile, args.size)