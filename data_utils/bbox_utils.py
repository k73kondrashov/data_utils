import numpy as np

def yolo2corners(yolo_boxes, w, h):
    yolo_boxes = check_dims(yolo_boxes)
    boxes = relative_to_abs(yolo_boxes, w, h)
    boxes = xywh2xyxy(boxes)
    return boxes

def corners2yolo(boxes, w, h):
    boxes = check_dims(boxes)
    yolo_boxes = xyxy2xywh(boxes)
    yolo_boxes = abs_to_relative(yolo_boxes, w, h)
    return yolo_boxes


def xyxy2xywh(xy_boxes):
    xy_boxes = check_dims(xy_boxes)
    wh_boxes = np.empty_like(xy_boxes)
    wh_boxes[:, :2] = (xy_boxes[:, :2] + xy_boxes[:, 2:]) / 2
    wh_boxes[:, 2:] = (xy_boxes[:, 2:] - xy_boxes[:, :2])
    return wh_boxes


def xywh2xyxy(wh_boxes):
    wh_boxes = check_dims(wh_boxes)
    xy_boxes = np.empty_like(wh_boxes)
    xy_boxes[:, :2] = wh_boxes[:, :2] - wh_boxes[:, 2:] / 2
    xy_boxes[:, 2:] = wh_boxes[:, :2] + wh_boxes[:, 2:] / 2
    return xy_boxes


def relative_to_abs(boxes, w, h):
    boxes = check_dims(boxes)
    boxes[:, ::2] *= w
    boxes[:, 1::2] *= h
    return boxes


def abs_to_relative(boxes, w, h):
    boxes = check_dims(boxes)
    boxes = boxes.astype(np.float32)
    boxes[:, ::2] /= w
    boxes[:, 1::2] /= h
    return boxes

def add_pad(boxes, pad_w, pad_h, img_w, img_h):
    boxes = check_dims(boxes)
    # TODO if all box out of image
    boxes[:, ::2] = np.clip(boxes[:, ::2] + pad_w, 0, img_w)
    boxes[:, 1::2] = np.clip(boxes[:, 1::2] + pad_h, 0, img_h)

    return boxes

def insert_boxes(boxes, crop, pad_xy):
    x1, y1, x2, y2 = crop
    pad_x, pad_y = pad_xy
    boxes[:, ::2] = np.clip(boxes[:, ::2], x1, x2)
    boxes[:, 1::2] = np.clip(boxes[:, 1::2], y1, y2)
    boxes[:, ::2] = boxes[:, ::2] - x1 + pad_x
    boxes[:, 1::2] = boxes[:, 1::2] - y1 + pad_y
    return boxes

def check_dims(x):
    x = np.array(x)
    if len(x.shape) == 1:
        x = np.expand_dims(x, 0)
    return x
