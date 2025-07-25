# # Entry point for Gradio app
# import os
# import sys
# import json
# import subprocess
# import threading
# import time
# import gradio as gr
# from config import STREAM_URL, MODEL_PATH
# from logger import logger
# from yolovision.streamer import start_flask_server
# from yolovision.detection import start_yolo_detection
# from yolovision.state import shared_state
# from flask import Flask

# # Globals
# stream_process = None
# flask_started = False


# def load_saved_user_info():
#     try:
#         with open("user_info.json", "r") as f:
#             return json.load(f)
#     except:
#         return None


# def save_user_info(name, password, link, name2=None, password2=None, link2=None):
#     with open("user_info.json", "w") as f:
#         json.dump({
#             "name": name,
#             "password": password,
#             "link": link,
#             "name2": name2 or "",
#             "password2": password2 or "",
#             "link2": link2 or ""
#         }, f)


# def generate_html(show_original, show_processed):
#     if not show_original and not show_processed:
#         return "<div style='text-align:center; font-size:20px;'>No streams selected.</div>"

#     html = "<div style='display:flex; justify-content:center; gap:20px; padding:20px;'>"
#     if show_original:
#         html += f"""
#         <div>
#             <h3 style="text-align:center;">Original Stream</h3>
#             <img src="{STREAM_URL}" style="border:2px solid #555; width:600px;" />
#         </div>"""
#     if show_processed:
#         html += f"""
#         <div>
#             <h3 style="text-align:center;">Processed Stream</h3>
#             <img src="http://localhost:7861/processed" style="border:2px solid #555; width:600px;" />
#         </div>"""
#     html += "</div>"
#     return html


# def check_video_status():
#     import cv2
#     try:
#         cap = cv2.VideoCapture(STREAM_URL)
#         if cap.isOpened():
#             ret, _ = cap.read()
#             cap.release()
#             if ret:
#                 return (
#                     "<div style='color:white;background:green;padding:10px;text-align:center;'>🟢 LIVE</div>",
#                     gr.update(interactive=False)
#                 )
#     except Exception as e:
#         logger.warning(f"Stream check failed: {e}")

#     return (
#         "<div style='color:white;background:red;padding:10px;text-align:center;'>🔴 DISCONNECTED</div>",
#         gr.update(interactive=True)
#     )


# def update_stats():
#     return str(shared_state["counter"]), "\n".join(shared_state["logs"][-10:])

 

# def handle_info_submission(name, password, link, name2, password2, link2):
#     global stream_process, flask_started

#     if not name or not password or not link:
#         return gr.update(visible=True), "❗ Fill all fields", gr.update(visible=False)

#     save_user_info(name, password, link, name2, password2, link2)
#     logger.info(f"Cam 1 Stream link: {link}")
#     if link2:
#         logger.info(f"Cam 2 Stream link: {link2}")

#     if stream_process is None or stream_process.poll() is not None:
#         logger.info("Starting stream_handler.py")
#         stream_process = subprocess.Popen(
#         [sys.executable, "stream_handler.py"],
#         stdout=subprocess.PIPE,
#         stderr=subprocess.PIPE,
#         creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
#         )

#         # Log its output in a thread so we can see what's going wrong
#         def log_stream_output(process):
#             for line in process.stdout:
#                 print("[stream_handler]", line.decode().strip())
#             for line in process.stderr:
#                 print("[stream_handler][ERR]", line.decode().strip())

#         threading.Thread(target=log_stream_output, args=(stream_process,), daemon=True).start()

#         time.sleep(2)

#     if not flask_started:
#         threading.Thread(target=start_flask_server, daemon=True).start()
#         flask_started = True
#         logger.info("Started Flask streamer")

#     threading.Thread(target=start_yolo_detection, daemon=True).start()
#     logger.info("Started YOLO detection")
#     return (
#         gr.update(visible=False),         # hide login
#         "✅ Started!",                    # show message
#         gr.update(visible=True),         # show video_section (Cam 1)
#         gr.update(visible=(link2 and link2.strip().lower() != "none"))  # show Cam 2 section only if link2 is valid
#     )



# def restart_app():
#     global stream_process
#     if stream_process and stream_process.poll() is None:
#         stream_process.terminate()
#         stream_process.wait()

#     python = sys.executable
#     script = os.path.abspath(__file__)
#     subprocess.Popen([python, script], creationflags=0x00000008)
#     os._exit(0)


# VALID_USERNAME = "0"
# VALID_PASSWORD = "0"

# def login_handler(username, password):
#     if username == VALID_USERNAME and password == VALID_PASSWORD:
#         logger.info("Login successful")
#         return gr.update(visible=False), gr.update(visible=True), ""
#     logger.warning("Login failed")
#     return gr.update(visible=True), gr.update(visible=False), "❌ Invalid credentials"


# # === Gradio Interface ===
# with gr.Blocks(title="Cotton Bale Detection") as demo:
#     gr.HTML("""
#     <style>
#     /* Hide footer, API button, logo, and settings */
#     footer, 
#     footer * ,
#     a[href^="https://gradio.app"],
#     a[href^="/settings"],
#     .gradio-container > div:nth-child(2) > div:nth-child(2),
#     div.svelte-1ipelgc { display: none !important; }

#     /* Also hide API / Settings nav elements if still shown */
#     a[href="/settings"], 
#     a[href="/api"], 
#     a[href^="https://www.gradio.app"] {
#         display: none !important;
#     }

#     /* Remove padding to clean leftover space */
#     .gradio-container { padding-bottom: 0px !important; }
#     </style>
#     """)

#     saved_user = load_saved_user_info()
#     # saved_user = None

#     login_section = gr.Column(visible=(saved_user.get("link", "").lower() is None))
#     user_info_section = gr.Column(visible=(saved_user is not None and saved_user.get("link", "").lower() != "none"))

#     video_section = gr.Column(visible=False)
#     video_section_cam2 = gr.Column(visible=False)

#     # gr.Markdown("## 🔐 Login")

#     with login_section:
#         username = gr.Textbox(label="Username")
#         password = gr.Textbox(label="Password", type="password")
#         login_btn = gr.Button("🔓 Login")
#         login_msg = gr.Markdown()

#     with user_info_section:
#         gr.Markdown("### 🧠 Enter Stream Info")
#         gr.Markdown("#### 🎥 Camera 1")
#         name = gr.Textbox(label="Name", value=(saved_user["name"] if saved_user else ""))
#         pwd = gr.Textbox(label="Password", value=(saved_user["password"] if saved_user else ""))
#         link = gr.Textbox(label="Stream URL", value=(saved_user["link"] if saved_user else ""))
#         gr.Markdown("#### 🎥 Camera 2 (Optional)")
#         name2 = gr.Textbox(label="Name", value=(saved_user["name2"] if saved_user else ""))
#         pwd2 = gr.Textbox(label="Password", value=(saved_user["password2"] if saved_user else ""))
#         link2 = gr.Textbox(label="Stream URL", value=(saved_user["link2"] if saved_user else ""))
#         submit_btn = gr.Button("✅ Submit")
#         info_msg = gr.Markdown()

#     with video_section:
#         gr.Markdown("## 🎥 Video Viewer Cam 1")
#         show_orig = gr.Checkbox(label="Show Original", value=False)
#         show_proc = gr.Checkbox(label="Show Processed", value=False)
#         restart_btn = gr.Button("🔄 Restart App", interactive=False)
#         video_html = gr.HTML()
#         status_box = gr.HTML()
#         count = gr.Textbox(label="Count", interactive=False)
#         logs = gr.Textbox(label="Logs", lines=10, interactive=False)

#         show_orig.change(generate_html, [show_orig, show_proc], video_html)
#         show_proc.change(generate_html, [show_orig, show_proc], video_html)

#         gr.Timer(1.0, "status").tick(fn=check_video_status, outputs=[status_box, restart_btn])
#         gr.Timer(1.0, "count").tick(fn=update_stats, outputs=[count, logs])

#     with gr.Column(visible=False) as video_section_cam2:
#         gr.Markdown("## 🎥 Video Viewer Cam 2")
#         show_orig2 = gr.Checkbox(label="Show Original", value=False)
#         show_proc2 = gr.Checkbox(label="Show Processed", value=False)
#         video_html2 = gr.HTML()
#         status_box2 = gr.HTML()
#         count2 = gr.Textbox(label="Count", interactive=False)
#         logs2 = gr.Textbox(label="Logs", lines=10, interactive=False)

#         show_orig2.change(generate_html, [show_orig2, show_proc2], video_html2)
#         show_proc2.change(generate_html, [show_orig2, show_proc2], video_html2)

#         gr.Timer(1.0, "status2").tick(fn=check_video_status, outputs=[status_box2, restart_btn])
#         gr.Timer(1.0, "count2").tick(fn=update_stats, outputs=[count2, logs2])

#     login_btn.click(login_handler, inputs=[username, password], outputs=[login_section, user_info_section, login_msg])
#     submit_btn.click(
#         handle_info_submission,
#         inputs=[name, pwd, link, name2, pwd2, link2],
#         outputs=[user_info_section, info_msg, video_section, video_section_cam2]
#     )
#     restart_btn.click(fn=restart_app, js="() => setTimeout(() => location.reload(), 11000)")

# demo.launch(server_port=7860)

# Entry point for Gradio app
import os
import sys
import json
import subprocess
import threading
import time
import gradio as gr
from config import STREAM_URL, MODEL_PATH
from logger import logger
from yolovision.streamer import start_flask_server
from yolovision.detection import start_yolo_detection
from yolovision.state import shared_state
from flask import Flask

# Globals
stream_process = None
flask_started = False

def load_saved_user_info():
    try:
        with open("user_info.json", "r") as f:
            return json.load(f)
    except:
        return None

def save_user_info(name, password, link):
    # First try to read existing data
    try:
        with open("user_info.json", "r") as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        # If file doesn't exist or is empty/invalid, create new structure
        data = {
            "name": "",  # or whatever default you want
            "password": "",
            "link": "",
            "name2": "",
            "password2": "",
            "link2": ""
        }
    
    # Update only the second user's info
    data["name"] = name
    data["password"] = password
    data["link"] = link
    
    # Write back to file
    with open("user_info.json", "w") as f:
        json.dump(data, f)

def generate_html(show_original, show_processed):
    if not show_original and not show_processed:
        return "<div style='text-align:center; font-size:20px;'>No streams selected.</div>"

    html = "<div style='display:flex; justify-content:center; gap:20px; padding:20px;'>"
    if show_original:
        html += f"""
        <div>
            <h3 style="text-align:center;">Original Stream</h3>
            <img src="{STREAM_URL}" style="border:2px solid #555; width:600px;" />
        </div>"""
    if show_processed:
        html += f"""
        <div>
            <h3 style="text-align:center;">Processed Stream</h3>
            <img src="http://localhost:7861/processed" style="border:2px solid #555; width:600px;" />
        </div>"""
    html += "</div>"
    return html

def check_video_status():
    import cv2
    try:
        cap = cv2.VideoCapture(STREAM_URL)
        if cap.isOpened():
            ret, _ = cap.read()
            cap.release()
            if ret:
                return (
                    "<div style='color:white;background:green;padding:10px;text-align:center;'>🟢 LIVE</div>",
                    gr.update(interactive=False)
                )
    except Exception as e:
        logger.warning(f"Stream check failed: {e}")

    return (
        "<div style='color:white;background:red;padding:10px;text-align:center;'>🔴 DISCONNECTED</div>",
        gr.update(interactive=True)
    )

def update_stats():
    return str(shared_state["counter"]), "\n".join(shared_state["logs"][-10:])

def handle_info_submission(name, password, link, *args):
    global stream_process, flask_started

    if not name or not password or not link:
        return gr.update(visible=True), "❗ Fill all fields", gr.update(visible=False)

    save_user_info(name, password, link)
    logger.info(f"Cam 1 Stream link: {link}")

    if stream_process is None or stream_process.poll() is not None:
        logger.info("Starting stream_handler.py")
        stream_process = subprocess.Popen(
            [sys.executable, "z.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
        )

        # Log its output in a thread so we can see what's going wrong
        def log_stream_output(process):
            for line in process.stdout:
                print("[stream_handler]", line.decode().strip())
            for line in process.stderr:
                print("[stream_handler][ERR]", line.decode().strip())

        threading.Thread(target=log_stream_output, args=(stream_process,), daemon=True).start()
        time.sleep(2)

    if not flask_started:
        threading.Thread(target=start_flask_server, daemon=True).start()
        flask_started = True
        logger.info("Started Flask streamer")

    threading.Thread(target=start_yolo_detection, daemon=True).start()
    logger.info("Started YOLO detection")
    return (
        gr.update(visible=False),         # hide login
        "✅ Started!",                    # show message
        gr.update(visible=True)           # show video_section (Cam 1)
    )

def restart_app():
    global stream_process
    if stream_process and stream_process.poll() is None:
        stream_process.terminate()
        stream_process.wait()

    python = sys.executable
    script = os.path.abspath(__file__)
    subprocess.Popen([python, script], creationflags=0x00000008)
    os._exit(0)

VALID_USERNAME = "0"
VALID_PASSWORD = "0"

def login_handler(username, password):
    if username == VALID_USERNAME and password == VALID_PASSWORD:
        logger.info("Login successful")
        return gr.update(visible=False), gr.update(visible=True), ""
    logger.warning("Login failed")
    return gr.update(visible=True), gr.update(visible=False), "❌ Invalid credentials"

# === Gradio Interface ===
with gr.Blocks(title="CAM1-Cotton Bale Detection") as demo:
    gr.HTML("""
    <style>
    /* Hide footer, API button, logo, and settings */
    footer, 
    footer * ,
    a[href^="https://gradio.app"],
    a[href^="/settings"],
    .gradio-container > div:nth-child(2) > div:nth-child(2),
    div.svelte-1ipelgc { display: none !important; }

    /* Also hide API / Settings nav elements if still shown */
    a[href="/settings"], 
    a[href="/api"], 
    a[href^="https://www.gradio.app"] {
        display: none !important;
    }

    /* Remove padding to clean leftover space */
    .gradio-container { padding-bottom: 0px !important; }
    </style>
    """)

    saved_user = load_saved_user_info()
    # saved_user = None

    login_section = gr.Column(visible=(saved_user.get("link", "").lower() is None))
    user_info_section = gr.Column(visible=(saved_user is not None and saved_user.get("link", "").lower() != "none"))

    video_section = gr.Column(visible=False)

    # gr.Markdown("## 🔐 Login")

    with login_section:
        username = gr.Textbox(label="Username")
        password = gr.Textbox(label="Password", type="password")
        login_btn = gr.Button("🔓 Login")
        login_msg = gr.Markdown()

    with user_info_section:
        gr.Markdown("### 🧠 Enter Stream Info for CAM 1")
        gr.Markdown("#### 🎥 Camera 1")
        name = gr.Textbox(label="Name", value=(saved_user["name"] if saved_user else ""))
        pwd = gr.Textbox(label="Password", value=(saved_user["password"] if saved_user else ""))
        link = gr.Textbox(label="Stream URL", value=(saved_user["link"] if saved_user else ""))
        submit_btn = gr.Button("✅ Submit")
        info_msg = gr.Markdown()

    with video_section:
        gr.Markdown("## 🎥 Video Viewer CAM 1")
        show_orig = gr.Checkbox(label="Show Original", value=False)
        show_proc = gr.Checkbox(label="Show Processed", value=False)
        restart_btn = gr.Button("🔄 Restart App", interactive=False)
        video_html = gr.HTML()
        status_box = gr.HTML()
        count = gr.Textbox(label="Count", interactive=False)
        logs = gr.Textbox(label="Logs", lines=10, interactive=False)

        show_orig.change(generate_html, [show_orig, show_proc], video_html)
        show_proc.change(generate_html, [show_orig, show_proc], video_html)

        gr.Timer(1.0, "status").tick(fn=check_video_status, outputs=[status_box, restart_btn])
        gr.Timer(1.0, "count").tick(fn=update_stats, outputs=[count, logs])

    login_btn.click(login_handler, inputs=[username, password], outputs=[login_section, user_info_section, login_msg])
    submit_btn.click(
        handle_info_submission,
        inputs=[name, pwd, link],
        outputs=[user_info_section, info_msg, video_section]
    )
    restart_btn.click(fn=restart_app, js="() => setTimeout(() => location.reload(), 11000)")

demo.launch(server_port=7860)
