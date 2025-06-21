import cv2
import numpy as np

aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)

# Marker for front face
marker_front = np.zeros((400, 400), dtype=np.uint8)
cv2.aruco.generateImageMarker(aruco_dict, 5, 400, marker_front, 1)
cv2.imwrite('l_marker_5.png', marker_front)

# Marker for side face
marker_side = np.zeros((400, 400), dtype=np.uint8)
cv2.aruco.generateImageMarker(aruco_dict, 15, 400, marker_side, 1)

cv2.imwrite('r_marker_15.png', marker_side)