import argparse
import os
import sys
import yaml
import cv2
import numpy as np
from loguru import logger
from typing import List, Dict

# Define valid input types for data classification
VALID_IMAGE_TYPES = ["i", "image"]
VALID_VIDEO_TYPES = ["v", "video"]

# Define valid file extensions for images and videos
IMAGE_EXT = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]
VIDEO_EXT = [".mp4", ".avi", ".mov", ".mkv", ".wmv"]

"""
Create command line argument parser
"""
def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="PreProcessing")
    parser.add_argument(
        "-t",
        "--type",
        type=str,
        default="image",
        choices=["image", "i", "video", "v"],
        help="Specify the data type: 'image(i)' or 'video(v)' (default: 'image')"
    )
    parser.add_argument(
        "-p",
        "--path",
        type=str,
        default="./assets",
        help="Path to images or video files (default: './assets/teddybear.jpeg')"
    )
    parser.add_argument(
        "-c",
        "--config",
        default="./config.yaml",
        type=str,
        help="Path to the configuration file (default: './config.yaml')"
    )
    parser.add_argument(
        "-o",
        "--output",
        default="./result",
        type=str,
        help="Path to the output directory (default: './result')"
    )
    return parser

"""
Load configuration settings from a yaml file
"""
def load_config(config_file_path: str) -> Dict:
    try:
        with open(config_file_path) as config_stream:
            return yaml.safe_load(config_stream)
    except Exception as e:
        logger.error(f"Error occurred while loading YAML: {e}")
        sys.exit(1)
    
"""
Create the output directory
"""
def create_output_directory(output_dir_path: str) -> None:
    if not os.path.exists(output_dir_path):
        os.makedirs(output_dir_path)
        logger.info(f"Created output directory: {output_dir_path}")
    else:
        logger.info(f"Output directory already exists: {output_dir_path}")

"""
Get a list of designated config files
"""
def get_supported_files(base_path: str, data_type: str) -> List[str]:
    extensions = IMAGE_EXT if data_type in VALID_IMAGE_TYPES else VIDEO_EXT

    supported_files = []
    try:
        if not os.path.exists(base_path):
            logger.warning(f"Specified path does not exist: {base_path}")
        else:
            if os.path.isfile(base_path) and os.path.splitext(base_path)[1].lower() in extensions:
                supported_files.append(base_path)
            elif os.path.isdir(base_path):
                for root_directory, _, filenames in os.walk(base_path):
                    for filename in filenames:
                        file_path = os.path.join(root_directory, filename)
                        if os.path.isfile(file_path) and os.path.splitext(file_path)[1].lower() in extensions:
                            supported_files.append(file_path)
            logger.info("Successfully retrieved supported file list.")
    except Exception as e:
        logger.error(f"Error occurred while searching supported files: {e}")
        sys.exit(1)
    return supported_files

"""
Cropping frame based on specified settings
"""
def set_cropped(frame: np.ndarray, input_file:str, settings: Dict) -> np.ndarray:
    if settings.get("enabled", True):
        s_x, s_y, e_x, e_y = settings.get("coordinates")
        if 0 <= s_x < e_x <= frame.shape[1] and 0 <= s_y < e_y <= frame.shape[0]:
            frame = frame[s_y:e_y, s_x:e_x].copy()
        else:
            logger.warning(f"Invalid coordinates for cropping in {input_file}. Skipping cropping.")
    return frame

"""
Resize the frame based on specified settings
"""
def set_resize(frame: np.ndarray, input_file:str, settings: Dict) -> np.ndarray:
    if settings.get("enabled", True):
        frame_resize = settings.get("output_size", frame.shape[:2][::-1])
        if frame_resize[0] > 0 and frame_resize[1] > 0:
            frame = cv2.resize(frame, tuple(frame_resize))
        else:
            logger.warning(f"Invalid resize dimensions for {input_file}. Skipping resizing.")
    return frame

"""
Adjust the brightness of the frame
"""
def set_brightness(frame: np.ndarray, input_file:str, settings: Dict) -> np.ndarray:
    if settings.get("enabled", True):
        frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        factor = settings.get("factor") if settings.get("enabled", True) else 1.0
        frame_hsv[:, :, 2] = np.clip(frame_hsv[:, :, 2] * factor, 0, 255)
        frame = cv2.cvtColor(frame_hsv, cv2.COLOR_HSV2BGR)
    return frame

"""
Adjust the saturation of the frame
"""
def set_saturation(frame: np.ndarray, input_file:str, settings: Dict) -> np.ndarray:
    if settings.get("enabled", True):
        frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        factor = settings.get("factor") if settings.get("enabled", True) else 1.0
        frame_hsv[:, :, 1] = np.clip(frame_hsv[:, :, 1] * factor, 0, 255)
        frame = cv2.cvtColor(frame_hsv, cv2.COLOR_HSV2BGR)
    return frame

"""
Rotate a given image frame by a specified angle
"""
def set_rotate_90(frame: np.ndarray, input_file:str, settings: Dict) -> np.ndarray:
    if settings.get("enabled", True):
        angle = settings.get("angle")
        if 0 < angle < 4:
            frame = np.rot90(frame, angle)
        else:
            logger.warning(f"Invalid dimensions for rotation in {input_file}. Skipping rotation.")
    return frame

"""
Flip the given frame either vertically or horizontally based on the specified option
"""
def set_flip(frame: np.ndarray, input_file:str, settings: Dict) -> np.ndarray:
    if settings.get("enabled", True):
        option = settings.get("options")
        if option in {'vertically', 'horizontally'}:
            frame = np.flipud(frame) if option == 'vertically' else np.fliplr(frame)
        else:
            logger.warning(f"Invalid flip option in {input_file}. Skipping flip.")
    return frame

"""
Process a single frame based on specified settings
"""
def process_frame(frame: np.ndarray, input_file:str, settings: Dict) -> np.ndarray:
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

"""
Preprocess multiple image or video files
"""
def preprocess_media(input_files: List[str], output_folder: str, settings: Dict) -> None:
    try:
        for input_file in input_files:
            sub_output_folder = os.path.abspath(os.path.join(output_folder, os.path.dirname(input_file)))
            create_output_directory(sub_output_folder)
            if input_file.lower().endswith(tuple(VIDEO_EXT)):
                video_process(input_file, sub_output_folder, settings)
            else:
                image_process(input_file, sub_output_folder, settings)
    except Exception as e:
        logger.error(f"Error occurred during media preprocessing: {e}")
        sys.exit(1)

"""
Main function to execute the preprocessing pipeline
"""
def main(args: argparse.Namespace) -> None:
    config_data = load_config(args.config)
    output_directory = args.output
    create_output_directory(output_directory)
    data_type = args.type
    input_path = args.path

    supported_files = get_supported_files(input_path, data_type)

    if supported_files:
        media_settings = config_data.get("image_settings", {}) if data_type in VALID_IMAGE_TYPES else config_data.get("video_settings", {})
        preprocess_media(supported_files, output_directory, media_settings)

if __name__ == "__main__":
    command_args = create_parser().parse_args()
    main(command_args)