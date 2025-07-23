import cv2
import time
import numpy as np
from datetime import datetime
from ultralytics import YOLO
from .state import shared_state2
from logger import logger
from config import *
from yolovision.utils import log_detection_to_csv
 
# In-memory track of objects for counting heuristics
tracked_objects = {}
 
# Will be set on the first frame
CENTER_LINE_Y = None
 
def wait_for_stream(url, timeout=15):
    start = time.time()
    while time.time() - start < timeout:
        cap = cv2.VideoCapture(url)
        if cap.isOpened():
            cap.release()
            return True
        time.sleep(1)
    return False
 
def log_event(object_id, event_type, anomaly_detected=False, anomaly_type=None):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    log_msg = f"{timestamp} - {event_type}"
    logger.info(log_msg)
    shared_state2["logs"].append(log_msg)
 
    line_count = shared_state2.get("counter", 0)
    obj = tracked_objects.get(object_id, {})
    pos_x = obj.get("current_x", "NA")
    pos_y = obj.get("current_y", "NA")
    counted = obj.get("counted", False)
    camera_id = "102"
    camera_name = "Camera 2"
 
    log_detection_to_csv(
        timestamp, object_id, event_type, line_count,
        pos_x, pos_y, counted, camera_id, camera_name,
        anomaly_detected, anomaly_type
    )
 
def start_yolo_detection2():
    global CENTER_LINE_Y
    logger.info("Starting YOLO detection thread (camera 2)")
    model = YOLO(MODEL_PATH)
 
    if not wait_for_stream(STREAM_URL2):
        logger.error("Stream not available after waiting; exiting detection thread.")
        return
 
    cap = cv2.VideoCapture(STREAM_URL2)
    if not cap.isOpened():
        logger.error("Failed to open video stream for camera 2")
        return
 
    first_frame = True
    cy_list = []
 
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            continue
 
        H, W, _ = frame.shape
 
        if first_frame:
            # horizontal center line half the frame height
            CENTER_LINE_Y = H // 2
            first_frame = False
 
        results = model.track(frame, persist=True, verbose=False)
 
        detections = []
        if results and results[0].boxes.id is not None:
            boxes   = results[0].boxes.xyxy.cpu().numpy().astype(int)
            ids     = results[0].boxes.id.cpu().numpy().astype(int)
            classes = results[0].boxes.cls.cpu().numpy()
            names   = results[0].names
 
            for (x1, y1, x2, y2), obj_id, cls_id in zip(boxes, ids, classes):
                label = names[int(cls_id)].lower()
                if label not in ("cottonbale", "coveredbale"):
                    continue
 
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)
 
                # initialize tracking record
                if obj_id not in tracked_objects:
                    started_side = "BOTTOM" if cy > CENTER_LINE_Y else "TOP"
                    tracked_objects[obj_id] = {
                        "first_y": cy,
                        "current_y": cy,
                        "started_side": started_side,
                        "counted": False
                    }
 
                obj = tracked_objects[obj_id]
                prev_y = obj["current_y"]
                obj["current_y"] = cy
                obj["current_x"] = cx  # store for logging
 
                # counting logic: require 5-frame downward trend
                if not obj["counted"]:
                    cy_list.append(cy)
                    if len(cy_list) == 5:
                        if all(y_prev > y_curr for y_prev, y_curr in zip(cy_list, cy_list[1:])):
                            if cy_list[0] - cy_list[-1] > 50:
                                shared_state2["counter"] += 1
                                obj["counted"] = True
                                event_name = "Covered Bale Detected" if label == "coveredbale" else "Bale Detected"
                                log_event(obj_id, f"{event_name}. Total: {shared_state2['counter']}")
                        cy_list.clear()
 
                # collect for overlay downstream
                detections.append({
                    "id": obj_id,
                    "label": label,
                    "box": (x1, y1, x2, y2),
                    "counted": obj["counted"]
                })
 
        # publish lightweight detection info
        # shared_state2["last_detections"] = detections
        # optionally publish raw frame for fallback
        shared_state2["last_raw_frame"]  = frame
 
        time.sleep(0.01)
 
    cap.release()
    logger.info("YOLO detection thread (camera 2) ended")
 