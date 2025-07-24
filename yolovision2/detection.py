import cv2
import time
import numpy as np
from datetime import datetime
from ultralytics import YOLO
from .state import shared_state2
from logger import logger
from config import *
from yolovision2.utils import log_detection_to_csv
 
tracked_objects = {}
 
# Simple center line for counting
CENTER_LINE_X = None
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
    log_msg = f"{timestamp} - Bale Detected"
    logger.info(log_msg)
    shared_state2["logs"].append(log_msg)
 
    line_count = shared_state2.get("counter", 0)
    obj = tracked_objects.get(object_id, {})
    pos_x = obj.get("current_x", "NA")
    pos_y = obj.get("current_y", "NA")  # Use this for vertical
    counted = obj.get("counted", False)
    camera_id = "101"
    camera_name = "Camera 1"
 
    log_detection_to_csv(
        timestamp, object_id, event_type, line_count,
        pos_x, pos_y, counted, camera_id, camera_name,
        anomaly_detected, anomaly_type
    )
 
def start_yolo_detection2():
    global CENTER_LINE_X
    logger.info("Starting YOLO detection thread")
    model = YOLO(MODEL_PATH)
 
    if not wait_for_stream(STREAM_URL2):
        logger.error("Stream not available after waiting. Exiting YOLO thread.")
        return
 
    cap = cv2.VideoCapture(STREAM_URL2)
    if not cap.isOpened():
        logger.error("Failed to open video stream")
        return
 
    # first_frame = True
    # cy_list = []
    # while cap.isOpened():
    #     ret, frame = cap.read()
    #     if not ret:
    #         continue
 
   
    #     H, W, _ = frame.shape
 
    #     # Set center line on first frame
    #     if first_frame:
    #         CENTER_LINE_X = W // 2
    #         CENTER_LINE_Y = H // 2  # Middle horizontal line (height-wise)
    #         first_frame = False
 
    #     results = model.track(frame, persist=True, verbose=False)
 
    #     if results[0].boxes.id is not None:
    #         boxes = results[0].boxes.xyxy.cpu().numpy()
    #         ids = results[0].boxes.id.cpu().numpy().astype(int)
    #         classes = results[0].boxes.cls.cpu().numpy()
    #         class_names = results[0].names
 
    #         for box, obj_id, cls_id in zip(boxes, ids, classes):
    #             name = class_names[int(cls_id)].lower()
    #             if name not in ["cottonbale", "coveredbale"]:
    #                 continue
 
    #             x1, y1, x2, y2 = box
    #             cy = int((x1 + x2) / 2)
    #             cy = int((y1 + y2) / 2)  # y-center
    #             cy = int(y1)
 
    #             # Initialize new objects
    #             if obj_id not in tracked_objects:
    #                 side = "BOTTOM" if cy > CENTER_LINE_Y else "TOP"
    #                 tracked_objects[obj_id] = {
    #                     "first_y": cy,
    #                     "current_y": cy,
    #                     "started_side": side,
    #                     "counted": False
    #                 }
 
    #             # Update position
    #             obj = tracked_objects[obj_id]
    #             prev_y = obj["current_y"]
    #             obj["current_y"] = cy
 
    #             # Check for crossing center line from bottom to top
    #             if not obj["counted"]:
    #                 # if prev_y > CENTER_LINE_Y and cy <= CENTER_LINE_Y:
    #                 #     shared_state2["counter"] += 1
    #                 #     obj["counted"] = True
    #                 #     if name == "coveredbale":
    #                 #         log_event(obj_id, f"Covered Bale Detected. Total: {shared_state2['counter']}",
    #                 #                 anomaly_detected=True, anomaly_type="Cotton Bale Wrap")
    #                 #     else:
    #                 #         log_event(obj_id, f"Bale Detected. Total: {shared_state2['counter']}")
    #                 cy_list.append(cy)
    #                 print(cy_list)
    #                 if len(cy_list) == 3:
    #                     print(cy_list)
    #                     is_decreasing = all(x > y for x, y in zip(cy_list, cy_list[1:]))
    #                     print("HEllo",is_decreasing)
    #                     if is_decreasing:
    #                         print("yes deacreasing")
    #                         check_sum = cy_list[0] - cy_list[2]
    #                         if check_sum > 50:
    #                             print("all_set")
    #                             shared_state2["counter"] += 1
    #                             obj["counted"] = True
    #                             log_event(obj_id, f"Bale Detected. Total: {shared_state2['counter']}")
    #                     cy_list = []
    first_frame = True
    temp_time = datetime.now()
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            continue

        H, W, _ = frame.shape
        
        # Set center line on first frame
        if first_frame:
            CENTER_LINE_X = (W // 2)-250  # Middle of frame for test i more move
            first_frame = False

        results = model.track(frame, persist=True, verbose=False)

        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            ids = results[0].boxes.id.cpu().numpy().astype(int)
            classes = results[0].boxes.cls.cpu().numpy()
            class_names = results[0].names

            for box, obj_id, cls_id in zip(boxes, ids, classes):
                name = class_names[int(cls_id)].lower()
                if name not in ["cottonbale", "coveredbale"]:
                    continue

                x1, y1, x2, y2 = box
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)

                # Initialize new objects
                if obj_id not in tracked_objects:
                    if datetime.now() == temp_time:
                        continue
                    if cy < 300:
                        continue
                    temp_time = datetime.now()
                    side = "RIGHT" if cy > CENTER_LINE_X else "LEFT"
                    tracked_objects[obj_id] = {
                        "first_x": cy,
                        "current_x": cy,
                        "current_y": cy,
                        "prev_x": cy,
                        "started_side": side,
                        "counted": False,
                        "cy_history": [cy]
                    }
                obj = tracked_objects[obj_id]
                obj["current_x"] = cy

                # Update per-object center history (max length 3)
                cy_history = obj.get("cy_history", [])
                cy_history.append(cy)
                if len(cy_history) > 3:
                    cy_history.pop(0)
                obj["cy_history"] = cy_history

                if not obj["counted"] and len(cy_history) == 3:
                    print(cy_history,obj)
                    is_decreasing = all(x > y for x, y in zip(cy_history, cy_history[1:]))
                    if is_decreasing:
                        check_sum = cy_history[0] - cy_history[2]
                        if check_sum > 15:
                            shared_state2["counter"] += 1
                            obj["counted"] = True
                            log_event(obj_id, f"Bale Detected. Total: {shared_state2['counter']}")
                # Draw bounding box
                if name == "coveredbale":
                    color = (0, 0, 255) if not obj["counted"] else (255, 0, 255)  # Red if not counted, magenta if counted
                else:
                    color = (0, 255, 0) if not obj["counted"] else (0, 255, 255)  # Green/yellow
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                cv2.circle(frame, (cy, cy), 5, (0, 0, 255), -1)
               
                # Labels
                status = "COUNTED" if obj["counted"] else f"Started {obj['started_side']}"
                cv2.putText(frame, f"ID: {obj_id}", (int(x1), int(y1)-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                cv2.putText(frame, f"x={cy} {status}", (int(x1), int(y2)+20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
 
        # Draw center counting line
        cv2.line(frame, (0, 300), (W, 300), (255, 0, 0), 3)
        cv2.putText(frame, "COTTON BALE DETECTION", (CENTER_LINE_X-70, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
       
        # Draw zones
        cv2.putText(frame, "IN ZONE", (CENTER_LINE_X + 50, H//2),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, "OUT ZONE", (CENTER_LINE_X + 50, 100),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
       
        # Direction arrow
        arrow_y = H - 50
        cv2.arrowedLine(frame, (CENTER_LINE_X + 100, arrow_y), (CENTER_LINE_X - 100, arrow_y),
                       (255, 255, 0), 3, tipLength=0.1)
        cv2.putText(frame, "OUT    IN", (CENTER_LINE_X - 60, arrow_y - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
 
        # Counter display
        cv2.rectangle(frame, (10, 10), (250, 80), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 10), (250, 80), (255, 255, 255), 2)
        cv2.putText(frame, f"COUNT: {shared_state2['counter']}", (20, 55),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3)
 
        shared_state2["last_frame"] = frame
        # time.sleep(0.01)
   
    cap.release()
    logger.info("YOLO detection thread ended")