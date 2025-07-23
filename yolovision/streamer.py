from flask import Flask, Response
import cv2
import threading
from yolovision.state import shared_state
from logger import logger
import numpy as np

app = Flask(__name__)

def generate_processed_stream():
    while True:
        frame = shared_state["last_frame"]
        if frame is not None:
            _, jpeg = cv2.imencode(".jpg", frame)
            frame_bytes = jpeg.tobytes()
            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")
        else:
            # If no frame is available, yield a black frame to keep the stream alive
            blank = 255 * np.ones((480, 640, 3), dtype=np.uint8)
            _, jpeg = cv2.imencode(".jpg", blank)
            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" + jpeg.tobytes() + b"\r\n")

@app.route('/processed')
def processed():
    return Response(generate_processed_stream(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

def start_flask_server():
    def run():
        logger.info("Flask MJPEG server running at http://localhost:7861/processed")
        app.run(host="0.0.0.0", port=7861, threaded=True)

    t = threading.Thread(target=run, daemon=True)
    t.start()
