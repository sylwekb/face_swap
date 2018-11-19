from itertools import count
import sys

import cv2
import dlib
import numpy as np

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


def warp_triangle(img1, img2, t1, t2):
    # https://www.learnopencv.com/warp-one-triangle-to-another-using-opencv-c-python/
    bb1 = cv2.boundingRect(np.float32([t1]))

    img1_cropped = img1[bb1[1] : bb1[1] + bb1[3], bb1[0] : bb1[0] + bb1[2]]

    bb2 = cv2.boundingRect(np.float32([t2]))

    t1_offset = [
        ((t1[0][0] - bb1[0]), (t1[0][1] - bb1[1])),
        ((t1[1][0] - bb1[0]), (t1[1][1] - bb1[1])),
        ((t1[2][0] - bb1[0]), (t1[2][1] - bb1[1])),
    ]
    t2_offset = [
        ((t2[0][0] - bb2[0]), (t2[0][1] - bb2[1])),
        ((t2[1][0] - bb2[0]), (t2[1][1] - bb2[1])),
        ((t2[2][0] - bb2[0]), (t2[2][1] - bb2[1])),
    ]
    mask = np.zeros((bb2[3], bb2[2], 3), dtype=np.float32)

    cv2.fillConvexPoly(mask, np.int32(t2_offset), (1.0, 1.0, 1.0), cv2.LINE_AA)

    size = (bb2[2], bb2[3])

    mat = cv2.getAffineTransform(np.float32(t1_offset), np.float32(t2_offset))

    img2_cropped = cv2.warpAffine(
        img1_cropped,
        mat,
        (size[0], size[1]),
        None,
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )

    img2_cropped = img2_cropped * mask

    img2_cropped_slice = np.index_exp[
        bb2[1] : bb2[1] + bb2[3], bb2[0] : bb2[0] + bb2[2]
    ]
    img2[img2_cropped_slice] = img2[img2_cropped_slice] * ((1.0, 1.0, 1.0) - mask)
    img2[img2_cropped_slice] = img2[img2_cropped_slice] + img2_cropped


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
points1 = detect_facial_landmarks(img1, FACE_DOWNSAMPLE_RATIO)

hull_index = cv2.convexHull(np.array(points1), returnPoints=False)

# Mouth points
# https://www.pyimagesearch.com/wp-content/uploads/2017/04/facial_landmarks_68markup-768x619.jpg
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
hull1 = [points1[hull_index_element[0]] for hull_index_element in hull_index]

rect = (0, 0, img1.shape[1], img1.shape[0])
delaunay_triangles = get_delaunay_triangles(rect, hull1, [hi[0] for hi in hull_index])

cap = cv2.VideoCapture(0)

for i in count(1):
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
        points2 = detect_facial_landmarks(img2, FACE_DOWNSAMPLE_RATIO)
    except Exception as e:
        print(e)
    else:
        hull2 = [points2[hull_index_element[0]] for hull_index_element in hull_index]

        img1_warped = np.float32(img2)

        for triangle in delaunay_triangles:
            mouth_points_set = set(mp[0] for mp in mouth_points)
            if (
                triangle[0] in mouth_points_set
                and triangle[1] in mouth_points_set
                and triangle[2] in mouth_points_set
            ):
                continue
            t1 = [points1[triangle[0]], points1[triangle[1]], points1[triangle[2]]]
            t2 = [points2[triangle[0]], points2[triangle[1]], points2[triangle[2]]]

            warp_triangle(img1, img1_warped, t1, t2)

        img2 = np.uint8(img1_warped)

    cv2.imshow("warp_triangle", img2)
    if cv2.waitKey(1) & 0xFF == 27:
        break
