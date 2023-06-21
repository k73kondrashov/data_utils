import numpy as np
from PIL import Image, ImageDraw

from bbox_utils import relative_to_abs


def boxes2mask(image, boxes, classes, colors, first_class_id=0):
    h, w = image.shape[:2]
    mask = Image.new('L', (w, h), 0)
    draw_mask = ImageDraw.Draw(mask)

    for class_id, color in enumerate(colors):
        target_boxes = boxes[classes == class_id]
        if len(target_boxes) == 0:
            continue

        for box in target_boxes:
            x1, y1, x2, y2 = box
            draw_mask.polygon(xy=[x1, y1, x2, y1, x2, y2, x1, y2], fill=(class_id + 1 - first_class_id))
    return np.array(mask)


def make_segment_mask(img, mask_points, mask_labels, target_classes, base_mask=None):
    h, w = img.shape[:2]
    mask = Image.new('L', (w, h), 0)
    draw_mask = ImageDraw.Draw(mask)

    for points, label in zip(mask_points, mask_labels):
        points = np.array(label[1:]).astype(float)
        points = relative_to_abs(points, w, h)[0]

        draw_mask.polygon(xy=list(points), fill=int(label))

    return np.array(mask)
