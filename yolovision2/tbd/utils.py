# Euclidean distance and helpers
import numpy as np

def euclidean(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

def is_cotton_near_man(cotton_box, man_boxes, threshold=100):
    cx = (cotton_box[0] + cotton_box[2]) / 2
    cy = (cotton_box[1] + cotton_box[3]) / 2
    return any(euclidean((cx, cy), ((x1 + x2)/2, (y1 + y2)/2)) < threshold for x1, y1, x2, y2 in man_boxes)
