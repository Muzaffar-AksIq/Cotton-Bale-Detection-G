# App configuration (paths, constants, etc.)
import os


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# print(BASE_DIR,"sssssssssssss")
MODEL_PATH = r"C:\Users\Asim\Desktop\CottonBale Project\model\BothScenerio.pt"
STREAM_URL = "http://localhost:9000/video"  # <--- this line must exist
STREAM_URL2 = "http://localhost:7000/video"  # <--- this line must exist
FLASK_PORT = 7861
GRADIO_PORT = 7860
LINE_IN_Y = 100
LINE_OUT_Y = 300
THRESHOLD = 0.5
