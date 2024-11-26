import cv2
import numpy as np
from card_utils import order_points, auto_rotate_points


def get_warped_card(frame, corners, width=500, height=700):
    ordered_corners = auto_rotate_points(corners)

    current_width = np.linalg.norm(ordered_corners[1] - ordered_corners[0])
    current_height = np.linalg.norm(ordered_corners[3] - ordered_corners[0])

    if current_width > current_height:
        width, height = height, width

    dst_points = np.array([
        [0, 0],
        [width - 1, 0],
        [width - 1, height - 1],
        [0, height - 1]
    ], dtype="float32")

    matrix = cv2.getPerspectiveTransform(ordered_corners, dst_points)
    warped = cv2.warpPerspective(frame, matrix, (width, height))

    # Convert warped image to binary (white card on black background)
    gray_warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    binary_warped = cv2.adaptiveThreshold(
        gray_warped,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11,
        2
    )

    # Memastikan kartu putih pada background hitam
    if np.mean(binary_warped[0:50, 0:50]) > 127:
        binary_warped = cv2.bitwise_not(binary_warped)

    return warped, binary_warped


def detect_card(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 50, 200)

    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(edged, kernel, iterations=1)

    contours, _ = cv2.findContours(
        dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_preview = np.zeros_like(edged)
    cv2.drawContours(contour_preview, contours, -1, (255, 255, 255), 2)

    if len(contours) > 0:
        card_contour = max(contours, key=cv2.contourArea)

        if cv2.contourArea(card_contour) > 5000:
            peri = cv2.arcLength(card_contour, True)
            approx = cv2.approxPolyDP(card_contour, 0.02 * peri, True)

            if len(approx) == 4:
                cv2.drawContours(frame, [approx], -1, (0, 255, 0), 2)
                corners = approx.reshape(4, 2)
                ordered_corners = order_points(corners)

                for corner in ordered_corners:
                    x, y = corner.astype(int)
                    cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)

                return True, ordered_corners, contour_preview, edged

    return False, None, contour_preview, edged
