import cv2
import dlib
import numpy as np
import random

RESIZE_HEIGHT = 360
FACE_DOWNSAMPLE_RATIO = 1.5

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")


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


def get_delaunay_triangles(rect, points, indexes):
    subdiv = cv2.Subdiv2D(rect)
    subdiv.insert(points)

    found_triangles = subdiv.getTriangleList()

    delaunay_triangles = []

    def contains(rect, point):
        return (
            rect[0] < point[0] < rect[0] + rect[2]
            and rect[1] < point[1] < rect[1] + rect[3]
        )

    for t in found_triangles:
        triangle = [(t[0], t[1]), (t[2], t[3]), (t[4], t[5])]

        # `getTriangleList` return triangles only, without origin points indices and we need them
        # so they correspond to other picture through index. So we're looking for original
        # index number for every point.
        if (
            contains(rect, triangle[0])
            and contains(rect, triangle[1])
            and contains(rect, triangle[2])
        ):

            indices = []
            for index, point in enumerate(points):
                if (
                    triangle[0][0] == point[0]
                    and triangle[0][1] == point[1]
                    or triangle[1][0] == point[0]
                    and triangle[1][1] == point[1]
                    or triangle[2][0] == point[0]
                    and triangle[2][1] == point[1]
                ):
                    indices.append(indexes[index])

                if len(indices) == 3:
                    delaunay_triangles.append(indices)
                    continue

    # remove duplicates
    return list(set(tuple(t) for t in delaunay_triangles))


cap = cv2.VideoCapture(0)
COLORS = []
first_frame = True
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
        points = detect_facial_landmarks(img2, FACE_DOWNSAMPLE_RATIO)
    except Exception:
        pass
    else:
        if first_frame:
            hull_index = cv2.convexHull(np.array(points), returnPoints=False)
            mouth_points = [
                # [48],  # <outer mouth>
                # [49],
                # [50],
                # [51],
                # [52],
                # [53],
                # [54],
                # [55],
                # [56],
                # [57],
                # [58],  # </outer mouth>
                [60],  # <inner mouth>
                [61],
                [62],
                [63],
                [64],
                [65],
                [66],
                [67],  # </inner mouth>
            ]
            hull_index = np.concatenate((hull_index, mouth_points))
            hull = [points[hull_index_element[0]] for hull_index_element in hull_index]

            mouth_points_set = set(mp[0] for mp in mouth_points)

            rect = (0, 0, img2.shape[1], img2.shape[0])
            delaunay_triangles = get_delaunay_triangles(
                rect, hull, [hi[0] for hi in hull_index]
            )
            # remove mouth points:
            delaunay_triangles[:] = [
                dt
                for dt in delaunay_triangles
                if not (
                    dt[0] in mouth_points_set
                    and dt[1] in mouth_points_set
                    and dt[2] in mouth_points_set
                )
            ]

            COLORS[:] = [
                (random.randint(0, 256), random.randint(0, 256), random.randint(0, 256))
                for _ in delaunay_triangles
            ]
            first_frame = False

        for color, triangle in zip(COLORS, delaunay_triangles):
            img2 = cv2.fillPoly(
                img2, [np.array([points[index] for index in triangle])], color
            )

    cv2.imshow("get_delaunay_triangles", img2)
    if cv2.waitKey(1) & 0xFF == 27:
        break
