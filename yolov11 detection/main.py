# main.py
import os
import numpy as np
from PIL import Image
from ultralytics import YOLO

BASE_DIR = os.path.dirname(os.path.abspath(__file__)) # __file__ refers to the current file's path
# BASE_DIR = "/path/to/your/models"  # Uncomment and set this if models are in a different directory

MODEL_CACHE = {}

ALL_MODELS = {
    "YOLOv11 Nano (Fastest)": "yolov11n.pt",
    "YOLOv11 Small": "yolov11s.pt",
    "YOLOv11 Medium": "yolov11m.pt",
    "YOLOv11 Large": "yolov11l.pt",
    "YOLOv11 X (Very Slow)": "yolov11x.pt",
}

# Absolute paths + existence check
AVAILABLE_MODELS = {
    "YOLOv11 Nano (Fastest)": "yolov11n.pt",
    "YOLOv11 Small": "yolov11s.pt",
    "YOLOv11 Medium": "yolov11m.pt",
    "YOLOv11 Large": "yolov11l.pt",
    "YOLOv11 X (Very Slow)": "yolov11x.pt",
}


def load_model(model_name):
    model_path = AVAILABLE_MODELS[model_name]

    if model_path not in MODEL_CACHE:
        MODEL_CACHE[model_path] = YOLO(model_path)

    return YOLO(MODEL_CACHE[model_path])


def detect_objects(image, model_name):
    if image is None:
        raise ValueError("No image uploaded")

    model = load_model(model_name)

    image = image.convert("RGB")
    image_np = np.array(image)

    results = model(image_np)
    return Image.fromarray(results[0].plot())
