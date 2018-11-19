import cv2
import numpy as np


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


img = cv2.imread("10c.jpg")


tri = [[360, 50], [60, 100], [300, 400]]

tri_out = [[400, 250], [160, 270], [400, 400]]

img_out = np.ones(img.shape, dtype=img.dtype) * 255
warp_triangle(img, img_out, tri, tri_out)

color = (0, 255, 0)

cv2.polylines(img, np.array([tri]), True, color, 2, cv2.LINE_AA)
cv2.polylines(img_out, np.array([tri_out]), True, color, 2, cv2.LINE_AA)


while True:
    cv2.imshow("image", img)
    if cv2.waitKey(0) & 0xFF == 27:  # esc
        break

    cv2.imshow("image", img_out)
    if cv2.waitKey(0) & 0xFF == 27:  # esc
        break
