from flask import Flask, render_template, request, send_file
import numpy as np
import cv2
from io import BytesIO
from core.calibrate import CalibrationSession

app = Flask(__name__)
session = CalibrationSession()

@app.route('/')
def index():
    return render_template('calibrate.html')

@app.route('/calibrate-frame', methods=['POST'])
def calibrate_frame():
    file = request.files['frame']
    img_array = np.frombuffer(file.read(), np.uint8)
    frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    detected, overlay = session.add_frame(frame)
    status_text = f"Frames: {len(session.objpoints)} / 15"

    if session.is_ready():
        session.calibrate('calibration_profile.npz')
        status_text += " | Calibrated ✔"

    # Draw status on overlay
    cv2.putText(overlay, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (0, 255, 0) if detected else (0, 0, 255), 2)

    # Return JPEG image
    _, img_encoded = cv2.imencode('.jpg', overlay)
    return BytesIO(img_encoded.tobytes()).getvalue(), 200, {'Content-Type': 'image/jpeg'}
