import cv2
import dlib
import numpy as np

RESIZE_HEIGHT = 360

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

cap = cv2.VideoCapture(0)

while True:
    ret, img = cap.read()
    if not ret:
        continue

    height, width = img.shape[:2]
    IMAGE_RESIZE = np.float32(height) / RESIZE_HEIGHT
    img = cv2.resize(
        img,
        None,
        fx=1.0 / IMAGE_RESIZE,
        fy=1.0 / IMAGE_RESIZE,
        interpolation=cv2.INTER_LINEAR,
    )
    try:
        det = max(detector(img), key=lambda r: r.area())
    except Exception as e:
        print(e)
    else:
        cv2.rectangle(
            img, (det.left(), det.top()), (det.right(), det.bottom()), (0, 0, 255), 2
        )

    cv2.imshow("detect face 0", img)
    if cv2.waitKey(1) & 0xFF == 27:
        break
