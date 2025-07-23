import cv2
import time
import numpy as np
from datetime import datetime
from ultralytics import YOLO
from .state import shared_state2
from logger import logger
from config import *

tracked_objects = {}

# Simple center line for counting
CENTER_LINE_X = None

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
    log_msg = f"{timestamp} - Bale Detected"
    logger.info(log_msg)
    shared_state2["logs"].append(log_msg)
    # print(log_msg)

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

    first_frame = True

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            continue

        H, W, _ = frame.shape
        
        # Set center line on first frame
        if first_frame:
            CENTER_LINE_X = W // 2  # Middle of frame
            # print(f"Frame size: {W}x{H}")
            # print(f"CENTER LINE at x={CENTER_LINE_X}")
            first_frame = False

        results = model.track(frame, persist=True, verbose=False)

        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            ids = results[0].boxes.id.cpu().numpy().astype(int)
            classes = results[0].boxes.cls.cpu().numpy()
            class_names = results[0].names

            for box, obj_id, cls_id in zip(boxes, ids, classes):
                name = class_names[int(cls_id)].lower()
                if name != "cottonbale":
                    continue

                x1, y1, x2, y2 = box
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)

                # Initialize new objects
                if obj_id not in tracked_objects:
                    side = "RIGHT" if cx > CENTER_LINE_X else "LEFT"
                    tracked_objects[obj_id] = {
                        "first_x": cx,
                        "current_x": cx,
                        "started_side": side,
                        "counted": False
                    }
                    # log_event(obj_id, f"NEW on {side} side at x={cx}")
                    # print(f"NEW Object {obj_id}: x={cx}, side={side}")
                
                # Update position
                obj = tracked_objects[obj_id]
                prev_x = obj["current_x"]
                obj["current_x"] = cx
                
                # Check for crossing center line from RIGHT to LEFT
                if not obj["counted"]:
                    # Object was on right, now on left
                    if prev_x > CENTER_LINE_X and cx <= CENTER_LINE_X:
                        shared_state2["counter"] += 1
                        obj["counted"] = True
                        log_event(obj_id, f"Bale Detected. Total: {shared_state2['counter']}")
                        # print(f">>> COUNTED Object {obj_id}! Total: {shared_state['counter']}")
                
                # Draw bounding box
                color = (0, 255, 0) if not obj["counted"] else (0, 255, 255)
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
                
                # Labels
                status = "COUNTED" if obj["counted"] else f"Started {obj['started_side']}"
                cv2.putText(frame, f"ID: {obj_id}", (int(x1), int(y1)-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                cv2.putText(frame, f"x={cx} {status}", (int(x1), int(y2)+20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        # Draw center counting line
        cv2.line(frame, (0, 300), (W, 300), (255, 0, 0), 3)
        cv2.putText(frame, "COTTON BALE DETECTION", (CENTER_LINE_X-70, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        # Draw zones
        cv2.putText(frame, "IN ZONE", (CENTER_LINE_X + 50, H//2), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, "OUT ZONE", (50, H//2), 
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
        time.sleep(0.01)
    
    cap.release()
    logger.info("YOLO detection thread ended")




# vertical line
# import cv2
# import time
# import numpy as np
# from datetime import datetime
# from ultralytics import YOLO
# from .state import shared_state2
# from logger import logger
# from config import *

# tracked_objects = {}

# # Simple center line for counting - now vertical
# CENTER_LINE_Y = None

# def wait_for_stream(url, timeout=15):
#     start = time.time()
#     while time.time() - start < timeout:
#         cap = cv2.VideoCapture(url)
#         if cap.isOpened():
#             cap.release()
#             return True
#         time.sleep(1)
#     return False

# def log_event(object_id, event_type):
#     timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
#     log_msg = f"{timestamp} - Bale Detected"
#     logger.info(log_msg)
#     shared_state2["logs"].append(log_msg)
#     # print(log_msg)

# def start_yolo_detection2():
#     global CENTER_LINE_Y
#     logger.info("Starting YOLO detection thread")
#     model = YOLO(MODEL_PATH)

#     if not wait_for_stream(STREAM_URL):
#         logger.error("Stream not available after waiting. Exiting YOLO thread.")
#         return

#     cap = cv2.VideoCapture(STREAM_URL)
#     if not cap.isOpened():
#         logger.error("Failed to open video stream")
#         return

#     first_frame = True

#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             continue

#         H, W, _ = frame.shape
        
#         # Set center line on first frame
#         if first_frame:
#             CENTER_LINE_Y = H // 2  # Middle of frame (vertical)
#             # print(f"Frame size: {W}x{H}")
#             # print(f"CENTER LINE at y={CENTER_LINE_Y}")
#             first_frame = False

#         results = model.track(frame, persist=True, verbose=False)

#         if results[0].boxes.id is not None:
#             boxes = results[0].boxes.xyxy.cpu().numpy()
#             ids = results[0].boxes.id.cpu().numpy().astype(int)
#             classes = results[0].boxes.cls.cpu().numpy()
#             class_names = results[0].names

#             for box, obj_id, cls_id in zip(boxes, ids, classes):
#                 name = class_names[int(cls_id)].lower()
#                 if name != "cottonbale":
#                     continue

#                 x1, y1, x2, y2 = box
#                 cx = int((x1 + x2) / 2)
#                 cy = int((y1 + y2) / 2)

#                 # Initialize new objects
#                 if obj_id not in tracked_objects:
#                     side = "BOTTOM" if cy > CENTER_LINE_Y else "TOP"
#                     tracked_objects[obj_id] = {
#                         "first_y": cy,
#                         "current_y": cy,
#                         "started_side": side,
#                         "counted": False
#                     }
#                     # log_event(obj_id, f"NEW on {side} side at y={cy}")
#                     # print(f"NEW Object {obj_id}: y={cy}, side={side}")
                
#                 # Update position
#                 obj = tracked_objects[obj_id]
#                 prev_y = obj["current_y"]
#                 obj["current_y"] = cy
                
#                 # Check for crossing center line from BOTTOM to TOP
#                 if not obj["counted"]:
#                     # Object was on bottom, now on top
#                     if prev_y > CENTER_LINE_Y and cy <= CENTER_LINE_Y:
#                         shared_state2["counter"] += 1
#                         obj["counted"] = True
#                         log_event(obj_id, f"Bale Detected. Total: {shared_state2['counter']}")
#                         # print(f">>> COUNTED Object {obj_id}! Total: {shared_state['counter']}")
                
#                 # Draw bounding box
#                 color = (0, 255, 0) if not obj["counted"] else (0, 255, 255)
#                 cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
#                 cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
                
#                 # Labels
#                 status = "COUNTED" if obj["counted"] else f"Started {obj['started_side']}"
#                 cv2.putText(frame, f"ID: {obj_id}", (int(x1), int(y1)-10), 
#                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
#                 cv2.putText(frame, f"y={cy} {status}", (int(x1), int(y2)+20), 
#                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

#         # Draw center counting line (horizontal line)
#         cv2.line(frame, (0, CENTER_LINE_Y), (W, CENTER_LINE_Y), (255, 0, 0), 3)
#         cv2.putText(frame, "COTTON BALE DETECTION", (W//2 - 120, CENTER_LINE_Y - 10), 
#                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
#         # Draw zones
#         cv2.putText(frame, "IN ZONE", (W//2 - 30, CENTER_LINE_Y - 50), 
#                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
#         cv2.putText(frame, "OUT ZONE", (W//2 - 40, CENTER_LINE_Y + 80), 
#                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
#         # Direction arrow (vertical)
#         arrow_x = W - 50
#         cv2.arrowedLine(frame, (arrow_x, CENTER_LINE_Y + 100), (arrow_x, CENTER_LINE_Y - 100), 
#                        (255, 255, 0), 3, tipLength=0.1)
#         cv2.putText(frame, "OUT", (arrow_x + 10, CENTER_LINE_Y + 80), 
#                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
#         cv2.putText(frame, "IN", (arrow_x + 10, CENTER_LINE_Y - 80), 
#                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

#         # Counter display
#         cv2.rectangle(frame, (10, 10), (250, 80), (0, 0, 0), -1)
#         cv2.rectangle(frame, (10, 10), (250, 80), (255, 255, 255), 2)
#         cv2.putText(frame, f"COUNT: {shared_state2['counter']}", (20, 55),
#                    cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3)

#         shared_state2["last_frame"] = frame
#         time.sleep(0.01)
    
#     cap.release()
#     logger.info("YOLO detection thread ended")