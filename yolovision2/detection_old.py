import cv2
import time
import numpy as np
from datetime import datetime
from ultralytics import YOLO
from .state import shared_state
from logger import logger
from config import *

tracked_objects = {}  # {id: {"first_seen": timestamp, "last_seen": timestamp, ...}}

# Line positions
entry_line_y = 200
exit_line_x = 1000
backup_entry_line_y = 220
backup_exit_line_x = 980

def wait_for_stream(url, timeout=15):
    start = time.time()
    while time.time() - start < timeout:
        cap = cv2.VideoCapture(url)
        if cap.isOpened():
            cap.release()
            return True
        time.sleep(1)
    return False

def log_event(object_id, event_type):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    logger.info(f"{timestamp} - Object {object_id} {event_type}")
    shared_state["logs"].append(timestamp)

def start_yolo_detection():
    logger.info("Starting YOLO detection thread")
    model = YOLO(MODEL_PATH)

    if not wait_for_stream(STREAM_URL):
        logger.error("Stream not available after waiting. Exiting YOLO thread.")
        return

    cap = cv2.VideoCapture(STREAM_URL)
    if not cap.isOpened():
        logger.error("Failed to open video stream")
        return

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            continue

        results = model.track(frame, persist=True, verbose=False)
        H, W, _ = frame.shape

        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            ids = results[0].boxes.id.cpu().numpy().astype(int)
            classes = results[0].boxes.cls.cpu().numpy()
            class_names = results[0].names

            for box, obj_id, cls_id in zip(boxes, ids, classes):
                name = class_names[int(cls_id)].lower()
                if name != "cottonbale":
                    print(name,"no track")
                    continue
                print("track")

                x1, y1, x2, y2 = box
                cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)

                if obj_id not in tracked_objects:
                    tracked_objects[obj_id] = {
                        "first_seen": datetime.now(),
                        "last_seen": datetime.now(),
                        "entry_line_crossed": False,
                        "exit_line_crossed": False,
                        "counted": False
                    }
                else:
                    tracked_objects[obj_id]["last_seen"] = datetime.now()

                # Entry logic
                if not tracked_objects[obj_id]["entry_line_crossed"]:
                    print("22")
                    if cy < entry_line_y:
                        tracked_objects[obj_id]["entry_line_crossed"] = True
                        log_event(obj_id, "CROSSED ENTRY LINE (Primary)")
                    elif cy < backup_entry_line_y:
                        tracked_objects[obj_id]["entry_line_crossed"] = True
                        log_event(obj_id, "CROSSED ENTRY LINE (Backup)")

                # Exit logic
                if tracked_objects[obj_id]["entry_line_crossed"] and not tracked_objects[obj_id]["exit_line_crossed"]:
                    if cx > exit_line_x:
                        tracked_objects[obj_id]["exit_line_crossed"] = True
                        if not tracked_objects[obj_id]["counted"]:
                            shared_state["counter"] += 1
                            tracked_objects[obj_id]["counted"] = True
                        log_event(obj_id, "CROSSED EXIT LINE (Primary)")
                    elif cx > backup_exit_line_x:
                        tracked_objects[obj_id]["exit_line_crossed"] = True
                        if not tracked_objects[obj_id]["counted"]:
                            shared_state["counter"] += 1
                            tracked_objects[obj_id]["counted"] = True
                        log_event(obj_id, "CROSSED EXIT LINE (Backup)")

                # Draw
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
                cv2.putText(frame, f"ID: {obj_id}", (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                print(shared_state)

        # Draw lines
        cv2.line(frame, (0, entry_line_y), (W, entry_line_y), (255, 0, 0), 2)
        cv2.line(frame, (exit_line_x, 0), (exit_line_x, H), (0, 255, 255), 2)
        cv2.line(frame, (0, backup_entry_line_y), (W, backup_entry_line_y), (255, 0, 255), 1)
        cv2.line(frame, (backup_exit_line_x, 0), (backup_exit_line_x, H), (0, 255, 255), 1)

        # Count text
        cv2.putText(frame, f"Total Count: {shared_state['counter']}", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

        shared_state["last_frame"] = frame
        time.sleep(0.01)
