import numpy as np

import cv2


def draw_boxes(image, boxes, color=(255, 0, 0), thick=2):
    for box in np.array(boxes).astype(int):
        x1, y1, x2, y2 = box
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thick)
