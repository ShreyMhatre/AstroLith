import argparse
import json
import os
import cv2
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Camera calibration from video using checkerboard frames.")
    parser.add_argument('--video', required=True, help='Path to input video file')
    return parser.parse_args()


def load_config(video_path):
    base_name = os.path.splitext(video_path)[0]
    config_path = f"{base_name}.json"
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with open(config_path, 'r') as f:
        return json.load(f)


def laplacian_variance(gray):
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def mean_corner_distance(c1, c2):
    return np.mean(np.linalg.norm(c1 - c2, axis=2))


def main():
    args = parse_args()
    try:
        config = load_config(args.video)
    except FileNotFoundError as e:
        print(str(e))
        return

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"Error: Cannot open video {args.video}")
        return

    checkerboard_size = tuple(config['checkerboard_size'])
    square_size = config['square_size']
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    objp = np.zeros((np.prod(checkerboard_size), 3), np.float32)
    objp[:, :2] = np.indices(checkerboard_size).T.reshape(-1, 2)
    objp *= square_size

    objpoints = []
    imgpoints = []

    frame_count = 0
    accepted_count = 0
    last_corners = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % config['frame_skip'] != 0:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        sharpness = laplacian_variance(gray)
        if sharpness < config['min_laplacian_var']:
            continue

        found, corners = cv2.findChessboardCorners(gray, checkerboard_size, None)
        if not found:
            continue

        corners_subpix = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

        if last_corners is not None:
            displacement = mean_corner_distance(corners_subpix, last_corners)
            if displacement < config['min_corner_move']:
                continue

        objpoints.append(objp.copy())
        imgpoints.append(corners_subpix)
        last_corners = corners_subpix
        accepted_count += 1

        print(f"Accepted frame {frame_count} (sharpness={sharpness:.1f})")

        if accepted_count >= config['max_frames']:
            break

    cap.release()

    if len(objpoints) < 5:
        print("Error: Not enough valid frames for calibration.")
        return

    image_size = gray.shape[::-1]
    ret, camera_matrix, dist_coeffs, _, _ = cv2.calibrateCamera(
        objpoints, imgpoints, image_size, None, None)

    video_base = os.path.splitext(os.path.basename(args.video))[0]
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'assets', 'profiles')
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"{video_base}_profile.npz")
    np.savez(output_file, camera_matrix=camera_matrix, dist_coeffs=dist_coeffs)
    print(f"Calibration successful. Saved to {output_file}")


if __name__ == '__main__':
    main()
