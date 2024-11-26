import numpy as np
import cv2


def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect


def auto_rotate_points(points):
    ordered = order_points(points)

    # Hitung tinggi dan lebar
    height_left = np.linalg.norm(ordered[3] - ordered[0])
    height_right = np.linalg.norm(ordered[2] - ordered[1])
    height = max(height_left, height_right)

    width_top = np.linalg.norm(ordered[1] - ordered[0])
    width_bottom = np.linalg.norm(ordered[2] - ordered[3])
    width = max(width_top, width_bottom)

    if width > height:
        new_ordered = np.array([
            ordered[3],
            ordered[0],
            ordered[1],
            ordered[2]
        ])
        return new_ordered

    return ordered
