import cv2
import dlib
import numpy as np

RESIZE_HEIGHT = 360
FACE_DOWNSAMPLE_RATIO = 1.5

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

cap = cv2.VideoCapture(0)


def detect_facial_landmarks(img, FACE_DOWNSAMPLE_RATIO=1):
    small_img = cv2.resize(
        img,
        None,
        fx=1.0 / FACE_DOWNSAMPLE_RATIO,
        fy=1.0 / FACE_DOWNSAMPLE_RATIO,
        interpolation=cv2.INTER_LINEAR,
    )

    # use the biggest face
    rect = max(detector(small_img), key=lambda r: r.area())

    scaled_rect = dlib.rectangle(
        int(rect.left() * FACE_DOWNSAMPLE_RATIO),
        int(rect.top() * FACE_DOWNSAMPLE_RATIO),
        int(rect.right() * FACE_DOWNSAMPLE_RATIO),
        int(rect.bottom() * FACE_DOWNSAMPLE_RATIO),
    )
    landmarks = predictor(img, scaled_rect)

    return [(point.x, point.y) for point in landmarks.parts()]


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
        points = detect_facial_landmarks(img, FACE_DOWNSAMPLE_RATIO)
    except Exception as e:
        print(e)
    else:
        hull_points = cv2.convexHull(np.array(points))
        img = cv2.fillPoly(img, [hull_points], (255, 0, 0))

    cv2.imshow("convexHull", img)
    if cv2.waitKey(1) & 0xFF == 27:
        break
