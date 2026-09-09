import time, json
from utils.files import save_image_with_meta
from settings import settings
from loguru import logger
from classes.Inference import Inference

def detect_age_gender_abnormal_data(inference: Inference, frame):
    # Create folder if it doesn't exist
    timestamp = str(round(time.time()))
    inferenced_data = inference.output()
    if inferenced_data is None: return None
    try:
        save_image_with_meta(settings.debug_image_folder, timestamp, frame, json.dumps(inferenced_data), max_files = settings.max_debug_images)
    except:
        pass

def detect_eye_tracking_abnormal_data(inference: Inference, frame):
    # Create folder if it doesn't exist
    timestamp = str(round(time.time()))
    inferenced_data = inference.output()
    try:
        save_image_with_meta(settings.debug_image_folder, timestamp, frame, json.dumps(inferenced_data), max_files = settings.max_debug_images)
    except:
        pass
