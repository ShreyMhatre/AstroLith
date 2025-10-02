## AstroLith

AstroLith is a Python-based image processing tool designed to calculate the volumetric weight of shipping containers and boxes using computer vision techniques. This project leverages aruco marker stickers and camera calibration to measure the physical dimensions of boxes, offering a contact-free, efficient alternative to traditional measurement tools.

### Project Aim
The main goal is to help shipping companies and logistics services measure the length, width, and height of customer-provided containers using images, especially when container dimensions are unknown. This facilitates accurate volumetric weight calculation, essential for determining shipping costs.

### How It Works
- **Aruco Markers:** Attach two aruco marker stickers perpendicular to each other on the box. These serve as reference points to measure dimensions.
- **Camera Calibration:** Before measurement, calibrate your camera using a chessboard matrix to capture intrinsic parameters. Calibration is a one-time process, and saved calibration files can be reused.
- **Measurement:** Capture an image of the container with aruco markers. The application detects the markers, calculates the dimensions, and computes the volumetric weight.

### Features
- No physical measurement tools required—uses computer vision and aruco markers.
- Device calibration for accurate measurements.
- Loads previously saved calibration files to save time.
- Designed for cross-device compatibility (works with different cameras via image input).
- Useful for logistics and shipping companies pricing by volumetric weight.

### Usage
1. **Calibrate your camera:**
   - Use a chessboard matrix and follow the instructions in the calibration module.
2. **Prepare your box:**
   - Stick 2 aruco marker stickers perpendicularly on the box.
3. **Capture and process the image:**
   - Run the main script, provide the calibration file, and an image of the container.
4. **Get volumetric weight:**
   - The program outputs the dimensions and calculated volumetric weight.

### Limitations
- Efficiency may vary depending on image quality and marker detection.
- Not an industrial solution, but a novel approach for quick estimates.
