import sys

import cv2
import dlib
import numpy as np

RESIZE_HEIGHT = 360

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

img1 = cv2.imread(sys.argv[1])
height, width = img1.shape[:2]
IMAGE_RESIZE = np.float32(height) / RESIZE_HEIGHT
img1 = cv2.resize(
    img1,
    None,
    fx=1.0 / IMAGE_RESIZE,
    fy=1.0 / IMAGE_RESIZE,
    interpolation=cv2.INTER_LINEAR,
)

det = max(detector(img1), key=lambda r: r.area())
img1_face = img1[det.top() : det.bottom(), det.left() : det.right()]

cap = cv2.VideoCapture(0)

while True:
    ret, img2 = cap.read()
    if not ret:
        continue

    height, width = img2.shape[:2]
    IMAGE_RESIZE = np.float32(height) / RESIZE_HEIGHT
    img2 = cv2.resize(
        img2,
        None,
        fx=1.0 / IMAGE_RESIZE,
        fy=1.0 / IMAGE_RESIZE,
        interpolation=cv2.INTER_LINEAR,
    )
    try:
        det = max(detector(img2), key=lambda r: r.area())
    except Exception as e:
        print(e)
    else:
        img2[det.top() : det.bottom(), det.left() : det.right()] = cv2.resize(
            img1_face, (det.right() - det.left(), det.bottom() - det.top())
        )
    cv2.imshow("detect face 0", img2)
    if cv2.waitKey(1) & 0xFF == 27:
        break
