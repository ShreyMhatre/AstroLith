# Camera Calibration from Video

This tool extracts high-quality checkerboard frames from a video to calibrate a camera using OpenCV. It requires a config file placed next to the video.

---

## 📁 File Requirements

- **Video file** (e.g., `session1.mp4`)
- **Config file** with the same base name (e.g., `session1.json`)

---

## ⚙️ `config.json` Format

```json
{
  "checkerboard_size": [9, 6],
  "square_size": 0.02,
  "min_corner_move": 30.0,
  "min_laplacian_var": 100.0,
  "frame_skip": 5,
  "max_frames": 20
}
```

### 🔍 Field Descriptions

| Field               | Type        | Description                                                            | Effect                                          |
|---------------------|-------------|------------------------------------------------------------------------|-------------------------------------------------|
| `checkerboard_size` | `[int, int]`| Number of inner corners `(columns, rows)` in the checkerboard         | Must match your printed calibration board       |
| `square_size`       | `float`     | Real-world size of one square in meters                                | Affects scale of output camera matrix           |
| `min_corner_move`   | `float`     | Minimum average pixel shift of corners to accept a new frame           | Filters out similar or redundant frames         |
| `min_laplacian_var` | `float`     | Minimum sharpness score (via Laplacian variance) to accept the frame   | Removes blurry or low-quality frames            |
| `frame_skip`        | `int`       | Process every Nth frame to reduce overhead                             | Lower value = more thorough but slower          |
| `max_frames`        | `int`       | Max number of good frames to collect for calibration                   | Higher = better accuracy but longer run         |

---

## 🧪 How to Run

```bash
python calibrate.py --video path/to/video.mp4
```

> The script will automatically look for a `path/to/video.json` file and exit with an error if it doesn't exist.

---

## 📌 Notes

- Ensure the checkerboard is fully visible, sharp, and seen from multiple angles.
- Use printed checkerboards with known square size (e.g., 20 mm).
- Do not include frames where the checkerboard is cut off or too far.
