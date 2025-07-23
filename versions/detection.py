# YOLO detection logic
import cv2
from ultralytics import YOLO
import time
from datetime import datetime
from .utils import euclidean, is_cotton_near_man
from .state import shared_state
from config import *
from logger import logger

import time

def wait_for_stream(url, timeout=15):
    import cv2
    start = time.time()
    while time.time() - start < timeout:
        cap = cv2.VideoCapture(url)
        if cap.isOpened():
            cap.release()
            return True
        time.sleep(1)
    return False


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

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        H, W, _ = frame.shape
        results = model(frame, verbose=False)[0]

        man_boxes, cotton_boxes = [], []

        for box in results.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = box
            if score < THRESHOLD:
                continue
            name = results.names[int(class_id)].lower()
            if name == "man":
                man_boxes.append((x1, y1, x2, y2))
            elif name == "cotton bale":
                cotton_boxes.append((x1, y1, x2, y2))

        # Update tracks and count logic...
        for x1, y1, x2, y2 in cotton_boxes:
            cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
            matched = False
            for obj_id, (px, py) in shared_state["tracked"].items():
                if euclidean((cx, cy), (px, py)) < 50:
                    shared_state["tracked"][obj_id] = (cx, cy)
                    shared_state["history"][obj_id].append(cy)
                    matched = True

                    if not shared_state["status"][obj_id]['near_man'] and is_cotton_near_man((x1, y1, x2, y2), man_boxes):
                        shared_state["status"][obj_id]['near_man'] = True

                    if not shared_state["status"][obj_id]['counted'] and any(y < LINE_IN_Y for y in shared_state["history"][obj_id]) and cy > LINE_OUT_Y:
                        shared_state["counter"] += 1
                        shared_state["status"][obj_id]['counted'] = True
                        shared_state["logs"].append(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                        logger.info(f"New cotton bale counted! Total: {shared_state['counter']}")
                    break

            if not matched:
                new_id = shared_state["next_id"]
                shared_state["tracked"][new_id] = (cx, cy)
                shared_state["history"][new_id] = [cy]
                shared_state["status"][new_id] = {
                    "counted": False,
                    "near_man": is_cotton_near_man((x1, y1, x2, y2), man_boxes)
                }
                shared_state["next_id"] += 1

        # Drawing
        for x1, y1, x2, y2 in cotton_boxes:
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 255, 0), 2)

        for x1, y1, x2, y2 in man_boxes:
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 2)

        cv2.line(frame, (int(W*0.2), LINE_IN_Y), (int(W*0.9), LINE_IN_Y), (255, 255, 0), 2)
        cv2.line(frame, (W - 90, int(H*0.2)), (W - 90, H), (0, 0, 255), 2)
        cv2.putText(frame, f"Total Count: {shared_state['counter']}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
        print(shared_state['counter'])

        shared_state["last_frame"] = frame
        time.sleep(0.01)
