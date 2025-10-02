import argparse
import os
import cv2
import numpy as np

def parse_args():
    """
    Parses command-line arguments for the camera calibration script.
    """
    parser = argparse.ArgumentParser(description="Camera calibration from video using a checkerboard pattern.")
    
    # Required Arguments
    parser.add_argument('--video', type=str, required=True, 
                        help='Path to the input video file for calibration.')
    parser.add_argument('--square_size', type=float, required=True, 
                        help='The side length of a single checkerboard square in meters.')

    # Optional Arguments with Defaults
    parser.add_argument('--checkerboard_size', type=int, nargs=2, default=[9, 6], 
                        metavar=('WIDTH', 'HEIGHT'),
                        help='Number of inner corners on the checkerboard (width height). Default: 9 6.')
    parser.add_argument('--frame_skip', type=int, default=12, 
                        help='Process every Nth frame in the video. Default: 12.')
    parser.add_argument('--sharpness_threshold', type=float, default=100.0, 
                        help='Minimum Laplacian variance to accept a frame (filters out blurry frames). Default: 100.0.')
    parser.add_argument('--min_corner_move', type=float, default=10.0, 
                        help='Minimum average pixel distance corners must move to accept a new frame. Default: 10.0.')
    parser.add_argument('--max_frames', type=int, default=120, 
                        help='Maximum number of valid frames to use for calibration. Default: 120.')

    return parser.parse_args()

def laplacian_variance(gray_image):
    """
    Calculates the variance of the Laplacian of a grayscale image, a measure of sharpness.
    """
    return cv2.Laplacian(gray_image, cv2.CV_64F).var()

def mean_corner_distance(corners1, corners2):
    """
    Calculates the mean Euclidean distance between two sets of corresponding corner points.
    """
    return np.mean(np.linalg.norm(corners1 - corners2, axis=2))

def main():
    """
    Main function to run the camera calibration process.
    """
    args = parse_args()

    # Open the video file
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"Error: Cannot open video file: {args.video}")
        return

    # Define checkerboard properties from arguments
    checkerboard_size = tuple(args.checkerboard_size)
    square_size = args.square_size
    
    # Termination criteria for corner refinement
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # Prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(8,5,0)
    objp = np.zeros((checkerboard_size[0] * checkerboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:checkerboard_size[0], 0:checkerboard_size[1]].T.reshape(-1, 2)
    objp *= square_size

    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3D points in real world space
    imgpoints = []  # 2D points in image plane.

    frame_count = 0
    accepted_count = 0
    last_corners = None

    print("Starting frame analysis... This may take a moment.")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % args.frame_skip != 0:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 1. Filter for sharpness
        sharpness = laplacian_variance(gray)
        if sharpness < args.sharpness_threshold:
            continue

        # 2. Find checkerboard corners
        found, corners = cv2.findChessboardCorners(gray, checkerboard_size, None)
        if not found:
            continue

        # Refine corner locations
        corners_subpix = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

        # 3. Filter for diversity (ensure board has moved)
        if last_corners is not None:
            displacement = mean_corner_distance(corners_subpix, last_corners)
            if displacement < args.min_corner_move:
                continue
        
        # If the frame is good, store the points
        objpoints.append(objp)
        imgpoints.append(corners_subpix)
        last_corners = corners_subpix
        accepted_count += 1

        print(f"Accepted frame {frame_count:4d} (Sharpness: {sharpness:.1f}, Kept: {accepted_count}/{args.max_frames})")

        if accepted_count >= args.max_frames:
            print("Maximum number of calibration frames reached.")
            break

    cap.release()

    if accepted_count < 10:
        print("\nError: Not enough valid frames for calibration.")
        print(f"Found only {accepted_count} valid frames. At least 10 are recommended.")
        print("Try adjusting the following parameters:")
        print(f"  --sharpness_threshold (current: {args.sharpness_threshold}): Lower this value if your video is slightly blurry.")
        print(f"  --min_corner_move (current: {args.min_corner_move}): Lower this if you moved the board slowly or in a small area.")
        print(f"  --frame_skip (current: {args.frame_skip}): Lower this to analyze more frames from your video.")
        return

    print(f"\nCalibrating using {len(objpoints)} valid frames...")

    # Perform camera calibration
    image_size = gray.shape[::-1]  # (width, height)
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, image_size, None, None
    )

    if not ret:
        print("Calibration failed. Could not compute camera parameters.")
        return

    # Create a directory for calibration profiles if it doesn't exist
    # This saves the file in 'assets/profiles/' relative to the script's parent directory
    script_dir = os.path.dirname(__file__)
    output_dir = os.path.join(os.path.dirname(script_dir), 'assets', 'profiles')
    os.makedirs(output_dir, exist_ok=True)
    
    video_base_name = os.path.splitext(os.path.basename(args.video))[0]
    output_file_path = os.path.join(output_dir, f"{video_base_name}_profile.npz")
    
    # Save the calibration results
    np.savez(output_file_path, 
             camera_matrix=camera_matrix, 
             dist_coeffs=dist_coeffs,
             image_size=image_size,
             square_size=square_size,
             checkerboard_size=checkerboard_size)
             
    print("\nCalibration successful!")
    print(f"Results saved to: {output_file_path}")

    # Optional: Print the results to the console
    print("\nCamera Matrix:")
    print(camera_matrix)
    print("\nDistortion Coefficients:")
    print(dist_coeffs)

if __name__ == '__main__':
    main()