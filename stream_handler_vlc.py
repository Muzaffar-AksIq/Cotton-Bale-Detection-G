# stream_handler_vlc.py

import asyncio
import json
import threading
import time
import cv2
import numpy as np
import vlc
from fastapi import FastAPI, Response
from fastapi.responses import StreamingResponse
from typing import AsyncGenerator
from contextlib import asynccontextmanager
import uvicorn

# Load user info
with open("user_info.json", "r") as f:
    data = json.load(f)

stream_link = data["link"]
username = data["name"]

print(f"Received Stream Link: {stream_link}")

# VLC-based streamer
class VLCStreamer:
    def __init__(self, rtsp_url: str, http_port: int = 8555):
        self.rtsp_url = rtsp_url
        self.http_port = http_port
        self.instance = vlc.Instance()
        self.player = self.instance.media_player_new()

        # Set VLC options to forward RTSP to HTTP
        options = (
            f':sout=#transcode{{vcodec=h264}}:std{{access=http,mux=ts,dst=127.0.0.1:{self.http_port}/stream}}',
            ':sout-keep',
            ':no-sout-all',
            ':http-host=0.0.0.0',
            ':network-caching=1000'
        )


        media = self.instance.media_new(self.rtsp_url, *options)
        self.player.set_media(media)

    def start(self):
        self.player.play()
        print("VLC player started for streaming...")
        time.sleep(5)  # Wait for VLC buffer to fill

    def stop(self):
        self.player.stop()

# Camera using OpenCV reading from localhost
class Camera:
    def __init__(self, url: str):
        self.cap = cv2.VideoCapture(url)
        self.lock = threading.Lock()

    def get_frame(self) -> bytes:
        with self.lock:
            ret, frame = self.cap.read()
            print(f"[Camera Read]: Success={ret}")
            if not ret:
                return b''
            ret, jpeg = cv2.imencode('.jpg', frame)
            return jpeg.tobytes() if ret else b''

    def release(self):
        with self.lock:
            if self.cap.isOpened():
                self.cap.release()

# FastAPI app with lifecycle
app = FastAPI()
vlc_streamer = VLCStreamer(stream_link)
camera = None  # defined at runtime

@asynccontextmanager
async def lifespan(app: FastAPI):
    global camera
    try:
        vlc_streamer.start()
        # print(f"http://127.0.0.1:{vlc_streamer.http_port}/stream")
        camera = Camera("http://127.0.0.1:8555/stream")
        yield
    finally:
        camera.release()
        vlc_streamer.stop()
        print("Camera and VLC resources released.")

app = FastAPI(lifespan=lifespan)

# Video stream generator
async def gen_frames() -> AsyncGenerator[bytes, None]:
    try:
        while True:
            frame = camera.get_frame()
            if frame:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            else:
                break
            await asyncio.sleep(1 / 15)
    except (asyncio.CancelledError, GeneratorExit):
        print("Frame generation cancelled.")
    finally:
        print("Frame generator exited.")

# Routes
@app.get("/video")
async def video_feed() -> StreamingResponse:
    return StreamingResponse(
        gen_frames(),
        media_type='multipart/x-mixed-replace; boundary=frame'
    )

@app.get("/snapshot")
async def snapshot() -> Response:
    frame = camera.get_frame()
    if frame:
        return Response(content=frame, media_type="image/jpeg")
    else:
        return Response(status_code=404, content="Camera frame not available.")

# Uvicorn entry
async def main():
    config = uvicorn.Config(app, host='0.0.0.0', port=9000)
    server = uvicorn.Server(config)
    await server.serve()

if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Server stopped by user.")
