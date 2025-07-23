import cv2
from ultralytics import YOLO
import time
from datetime import datetime
from .utils import euclidean
from .state import shared_state2
from config import *
from logger import logger

def wait_for_stream(url, timeout=15):
    start = time.time()
    while time.time() - start < timeout:
        cap = cv2.VideoCapture(url)
        if cap.isOpened():
            cap.release()
            return True
        time.sleep(1)
    return False

def start_yolo_detection2():
    logger.info("Starting YOLO detection thread")
    model = YOLO(MODEL_PATH)

    if not wait_for_stream(STREAM_URL2):
        logger.error("Stream not available after waiting. Exiting YOLO thread.")
        return

    cap = cv2.VideoCapture(STREAM_URL2)
    if not cap.isOpened():
        logger.error("Failed to open video stream")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        H, W, _ = frame.shape
        results = model(frame, verbose=False)[0]

        cotton_boxes = []

        for box in results.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = box
            if score < THRESHOLD:
                continue
            name = results.names[int(class_id)].lower()
            if name == "cottonbale":
                cotton_boxes.append((x1, y1, x2, y2))

        # Tracking and counting logic
        for x1, y1, x2, y2 in cotton_boxes:
            cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
            matched = False

            for obj_id, (px, py) in shared_state2["tracked"].items():
                if euclidean((cx, cy), (px, py)) < 80:  # increased match threshold
                    shared_state2["tracked"][obj_id] = (cx, cy)
                    shared_state2["history"][obj_id].append(cy)
                    matched = True

                    if not shared_state2["status"][obj_id]['counted'] and \
                    any(y < LINE_IN_Y for y in shared_state2["history"][obj_id]) and cy > LINE_OUT_Y:

                        # ✅ Time gap logic added here
                        time_since_last = datetime.now() - shared_state2["last_count_time"]
                        if time_since_last.total_seconds() > 1.5:  # Wait at least 1.5 seconds
                            shared_state2["counter"] += 1
                            shared_state2["status"][obj_id]["counted"] = True
                            shared_state2["last_count_time"] = datetime.now()  # update last count time
                            shared_state2["logs"].append(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                            logger.info(f"New cotton bale counted! Total: {shared_state2['counter']}")
                    break

            if not matched:
                # Avoid duplicate objects by checking existing ones again
                already_exists = False
                for obj_id, (px, py) in shared_state2["tracked"].items():
                    if euclidean((cx, cy), (px, py)) < 80:
                        already_exists = True
                        break

                if not already_exists:
                    new_id = shared_state2["next_id"]
                    shared_state2["tracked"][new_id] = (cx, cy)
                    shared_state2["history"][new_id] = [cy]

                    counted = False
                    if cy > LINE_OUT_Y:
                        time_since_last = datetime.now() - shared_state2["last_count_time"]
                        if time_since_last.total_seconds() > 1.5:
                            shared_state2["counter"] += 1
                            shared_state2["logs"].append(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                            logger.info(f"New cotton bale counted (post-line spawn)! Total: {shared_state2['counter']}")
                            shared_state2["last_count_time"] = datetime.now()
                            counted = True

                    shared_state2["status"][new_id] = {
                        "counted": counted
                    }
                    shared_state2["next_id"] += 1

        # Drawing
        for x1, y1, x2, y2 in cotton_boxes:
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 255, 0), 2)

        cv2.line(frame, (int(W*0.2), LINE_IN_Y), (int(W*0.9), LINE_IN_Y), (255, 255, 0), 2)
        cv2.line(frame, (W - 90, int(H*0.2)), (W - 90, H), (0, 0, 255), 2)
        cv2.putText(frame, f"Total Count: {shared_state2['counter']}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

        shared_state2["last_frame"] = frame
        time.sleep(0.01)
