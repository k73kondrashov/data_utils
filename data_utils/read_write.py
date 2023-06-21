import numpy as np


def read_yolo_labels(path):
    with open(path) as f:
        labels = np.array([line.rstrip().split() for line in f.readlines()])
    return labels
