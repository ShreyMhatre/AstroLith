import cv2
import numpy as np
import os
import argparse
import glob

def get_unique_filepath(directory, base_name, extension):
    """
    Generates a unique filepath by appending a number if the file already exists.
    """
    output_path = os.path.join(directory, f"{base_name}{extension}")
    counter = 1
    while os.path.exists(output_path):
        output_path = os.path.join(directory, f"{base_name}_{counter}{extension}")
        counter += 1
    return output_path

def assess_calibration_visually(camera_matrix, dist_coeffs, image_paths, output_dir="calibration_examples"):
    """
    Saves a few side-by-side images (original vs. undistorted) to visually assess calibration quality.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory for example images: '{output_dir}'")

    num_examples = min(len(image_paths), 4)
    if not image_paths or num_examples == 0:
        print("No images found to create visual examples.")
        return
        
    example_paths = np.random.choice(image_paths, num_examples, replace=False)

    for i, path in enumerate(example_paths):
        original_img = cv2.imread(path)
        if original_img is None: continue
        
        undistorted_img = cv2.undistort(original_img, camera_matrix, dist_coeffs, None, None)
        h, w, _ = original_img.shape
        
        if h > 0 and w > 0:
            comparison_img = cv2.hconcat([original_img, undistorted_img])
            cv2.putText(comparison_img, "Original (Distorted)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(comparison_img, "Corrected (Undistorted)", (w + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            output_path = os.path.join(output_dir, f"comparison_{i+1}.jpg")
            cv2.imwrite(output_path, comparison_img)
        
    print(f"\nSaved {num_examples} visual comparison images to the '{output_dir}' directory.")
    print("Check these images to confirm that distorted lines are now straight.")


def calibrate(image_dir, rows, cols, square_size, view_detections, save_examples):
    """
    Performs camera calibration using checkerboard images.
    """
    checkerboard_dims = (cols, rows)
    objp = np.zeros((checkerboard_dims[0] * checkerboard_dims[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:checkerboard_dims[0], 0:checkerboard_dims[1]].T.reshape(-1, 2)
    objp = objp * square_size

    objpoints = []
    imgpoints = []
    
    # ============================ FIX IS HERE ============================
    # Expanded search to include multiple common image formats.
    supported_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    image_paths = []
    for ext in supported_extensions:
        image_paths.extend(glob.glob(os.path.join(image_dir, ext)))
    image_paths = sorted(image_paths)
    # =====================================================================

    if not image_paths:
        print(f"Error: No supported images (.jpg, .jpeg, .png, .bmp) found in directory '{image_dir}'.")
        return None, None

    print(f"Found {len(image_paths)} images. Processing...")
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    gray_shape = None

    for i, image_path in enumerate(image_paths):
        img = cv2.imread(image_path)
        if img is None: continue
            
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if gray_shape is None: gray_shape = gray.shape[::-1]
            
        ret, corners = cv2.findChessboardCorners(gray, checkerboard_dims, None)

        if ret:
            print(f"  [{i+1}/{len(image_paths)}] Detected corners in {os.path.basename(image_path)}")
            objpoints.append(objp)
            corners_subpix = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners_subpix)

            if view_detections:
                cv2.drawChessboardCorners(img, checkerboard_dims, corners_subpix, ret)
                cv2.imshow('Corner Detections', img)
                cv2.waitKey(500)
        else:
            print(f"  [{i+1}/{len(image_paths)}] Could not detect corners in {os.path.basename(image_path)}. Skipping.")

    if view_detections: cv2.destroyAllWindows()
    if not objpoints:
        print("\nError: Could not find corners in ANY of the images. Calibration failed.")
        return None, None

    print("\n--- Performing Calibration ---")
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, gray_shape, None, None
    )

    if not ret:
        print("Calibration failed.")
        return None, None
    
    mean_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], camera_matrix, dist_coeffs)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        mean_error += error
    reprojection_error = mean_error / len(objpoints)

    print("\n--- Calibration Accuracy Report ---")
    print(f"Mean Re-projection Error: {reprojection_error:.4f} pixels")
    if reprojection_error < 0.5: print(" -> RESULT: Excellent. A high-quality calibration.")
    elif reprojection_error < 1.0: print(" -> RESULT: Good. Acceptable for most applications.")
    else: print(" -> RESULT: Needs Improvement. Consider re-taking photos.")

    if save_examples:
        assess_calibration_visually(camera_matrix, dist_coeffs, image_paths)

    return camera_matrix, dist_coeffs

def main():
    parser = argparse.ArgumentParser(
        description="Camera calibration using a checkerboard pattern. Defaults to a 9x6 board.",
        formatter_class=argparse.RawTextHelpFormatter)
        
    parser.add_argument('--dir', type=str, required=True, help="Path to the directory containing calibration images.")
    parser.add_argument('--size', type=float, required=True, help="The real-world size of a checkerboard square (e.g., 25 for 25mm).")
    parser.add_argument('--output', type=str, default=None, 
                        help="Optional: Full path to save the output .npz file.\n"
                             "If not provided, it saves in 'assets/profiles/' using the directory name.")
    parser.add_argument('--rows', type=int, default=5, help="Number of inner corners on the checkerboard's height (Default: 5).")
    parser.add_argument('--cols', type=int, default=8, help="Number of inner corners on the checkerboard's width (Default: 8).")
    parser.add_argument('--view', action='store_true', help="Optional: Display corner detections on each image.")
    parser.add_argument('--save_examples', action='store_true', help="Optional: Save 'before and after' images to check accuracy.")
    args = parser.parse_args()

    cam_matrix, dist_coeffs = calibrate(args.dir, args.rows, args.cols, args.size, args.view, args.save_examples)

    if cam_matrix is not None and dist_coeffs is not None:
        if args.output is None:
            save_dir = os.path.join("assets", "profiles")
            os.makedirs(save_dir, exist_ok=True)
            base_name = os.path.basename(os.path.normpath(args.dir))
            output_path = get_unique_filepath(save_dir, base_name, ".npz")
        else:
            output_path = args.output
            custom_dir = os.path.dirname(output_path)
            if custom_dir: os.makedirs(custom_dir, exist_ok=True)

        np.savez(output_path, camera_matrix=cam_matrix, dist_coeffs=dist_coeffs)
        print(f"\nCalibration successful. Profile saved to '{output_path}'")

if __name__ == '__main__':
    main()