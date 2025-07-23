# Euclidean distance and helpers
import numpy as np
import csv
import os

CSV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'detections_log.csv')
CSV_PATH = os.path.abspath(CSV_PATH) 

def log_detection_to_csv(
    timestamp, object_id, event_type, line_count, pos_x, pos_y, counted,
    camera_id, camera_name, anomaly_detected=False, anomaly_type=None
):
    file_exists = os.path.isfile(CSV_PATH)
    with open(CSV_PATH, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow([
                'Timestamp', 'CameraID', 'CameraName',
                'ObjectID', 'EventType', 'LineCount',
                'PosX', 'PosY', 'Counted',
                'AnomalyDetected', 'AnomalyType'
            ])
        writer.writerow([
            timestamp, camera_id, camera_name,
            object_id, event_type, line_count,
            pos_x, pos_y, counted,
            str(anomaly_detected).lower(),    # Will appear as 'true' or 'false'
            anomaly_type if anomaly_type is not None else ""
        ])


def euclidean(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

def is_cotton_near_man(cotton_box, man_boxes, threshold=100):
    cx = (cotton_box[0] + cotton_box[2]) / 2
    cy = (cotton_box[1] + cotton_box[3]) / 2
    return any(euclidean((cx, cy), ((x1 + x2)/2, (y1 + y2)/2)) < threshold for x1, y1, x2, y2 in man_boxes)
