# image_video_utils.py
import os
import cv2
import numpy as np
from loguru import logger
from typing import Dict
from utils.__env import *
from utils.file import create_output_directory

def set_cropped(frame: np.ndarray, input_file: str, settings: Dict) -> np.ndarray:
    if settings.get("enabled", True):
        s_x, s_y, e_x, e_y = settings.get("coordinates")
        if 0 <= s_x < e_x <= frame.shape[1] and 0 <= s_y < e_y <= frame.shape[0]:
            frame = frame[s_y:e_y, s_x:e_x].copy()
        else:
            logger.warning(f"Invalid coordinates for cropping in {input_file}. Skipping cropping.")
    return frame

def set_resize(frame: np.ndarray, input_file: str, settings: Dict) -> np.ndarray:
    if settings.get("enabled", True):
        frame_resize = settings.get("output_size", frame.shape[:2][::-1])
        if frame_resize[0] > 0 and frame_resize[1] > 0:
            frame = cv2.resize(frame, tuple(frame_resize))
        else:
            logger.warning(f"Invalid resize dimensions for {input_file}. Skipping resizing.")
    return frame

def set_brightness(frame: np.ndarray, input_file: str, settings: Dict) -> np.ndarray:
    if settings.get("enabled", True):
        frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        factor = settings.get("factor") if settings.get("enabled", True) else 1.0
        frame_hsv[:, :, 2] = np.clip(frame_hsv[:, :, 2] * factor, 0, 255)
        frame = cv2.cvtColor(frame_hsv, cv2.COLOR_HSV2BGR)
    return frame

def set_saturation(frame: np.ndarray, input_file: str, settings: Dict) -> np.ndarray:
    if settings.get("enabled", True):
        frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        factor = settings.get("factor") if settings.get("enabled", True) else 1.0
        frame_hsv[:, :, 1] = np.clip(frame_hsv[:, :, 1] * factor, 0, 255)
        frame = cv2.cvtColor(frame_hsv, cv2.COLOR_HSV2BGR)
    return frame

def set_rotate_90(frame: np.ndarray, input_file: str, settings: Dict) -> np.ndarray:
    if settings.get("enabled", True):
        angle = settings.get("angle")
        if 0 < angle < 4:
            frame = np.rot90(frame, angle)
        else:
            logger.warning(f"Invalid dimensions for rotation in {input_file}. Skipping rotation.")
    return frame

def set_flip(frame: np.ndarray, input_file: str, settings: Dict) -> np.ndarray:
    if settings.get("enabled", True):
        option = settings.get("options")
        if option in {'vertically', 'horizontally'}:
            frame = np.flipud(frame) if option == 'vertically' else np.fliplr(frame)
        else:
            logger.warning(f"Invalid flip option in {input_file}. Skipping flip.")
    return frame

"""
Preprocess a image frame
"""
def image_process(input_file: str, output_folder: str, settings: Dict) -> None:
    frame = cv2.imread(input_file, cv2.IMREAD_UNCHANGED)
    processed_frame = process_frame(frame, input_file, settings)
    output_path = os.path.join(output_folder, os.path.basename(input_file))
    cv2.imwrite(output_path, processed_frame)

"""
Preprocess a image frame
"""
def video_process(input_file: str, output_folder: str, settings: Dict) -> None:
    try:
        preview = settings.get("preview")
        save_video = settings.get("save_video")
        save_image = settings.get("save_image")

        cap = cv2.VideoCapture(input_file)
        if not cap.isOpened():
            logger.error(f"Unable to open the video file: {input_file}")
            return

        if save_video:
            output_directory = os.path.abspath(os.path.join(output_folder, os.path.splitext(os.path.basename(input_file))[0]))
            create_output_directory(output_directory)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            dummy_h, dummy_w, dummy_ch = process_frame(np.zeros((int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), 3), dtype=np.uint8), '', settings).shape
            out = cv2.VideoWriter(os.path.join(output_directory, os.path.basename(input_file)), fourcc, cap.get(cv2.CAP_PROP_FPS), (dummy_w, dummy_h))
        if save_image:
            output_directory_img = os.path.abspath(os.path.join(output_folder, os.path.splitext(os.path.basename(input_file))[0], "img"))
            create_output_directory(output_directory_img)

        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = process_frame(frame, input_file, settings)

            if preview:
                cv2.imshow(f'Preview - {os.path.basename(input_file)}', frame)
                cv2.waitKey(1)
            if save_video:
                out.write(frame)
            if save_image:
                save_path = os.path.join(output_directory_img, f"frame_{frame_count}.png")
                cv2.imwrite(save_path, frame)
            frame_count += 1

        if preview:
            cv2.destroyAllWindows()
        if save_video:
            out.release()
        cap.release()
    except Exception as e:
        logger.error(f"Error occurred during video preprocessing: {e}")
        sys.exit(1)
        
def process_frame(frame: np.ndarray, input_file: str, settings: Dict) -> np.ndarray:
    cropped_settings = settings.get("cropped", {})
    resize_settings = settings.get("resize", {})
    brightness_settings = settings.get("brightness", {})
    saturation_settings = settings.get("saturation", {})
    rotate_settings = settings.get("rotate", {})
    flip_settings = settings.get("flip", {})

    frame = set_cropped(frame, input_file, cropped_settings)
    frame = set_resize(frame, input_file, resize_settings)
    frame = set_brightness(frame, input_file, brightness_settings)
    frame = set_saturation(frame, input_file, saturation_settings)
    frame = set_rotate_90(frame, input_file, rotate_settings)
    frame = set_flip(frame, input_file, flip_settings)

    return frame