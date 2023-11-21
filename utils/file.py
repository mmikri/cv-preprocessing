# file_utils.py
import os
from loguru import logger
from typing import List
from dotenv import load_dotenv
from utils.__env import *

def create_output_directory(output_dir_path: str) -> None:
    if not os.path.exists(output_dir_path):
        os.makedirs(output_dir_path)
        logger.info(f"Created output directory: {output_dir_path}")
    else:
        logger.info(f"Output directory already exists: {output_dir_path}")

def get_supported_files(base_path: str, data_type: str) -> List[str]:
    extensions = IMAGE_EXT if data_type in VALID_IMAGE_TYPES else VIDEO_EXT
    supported_files = []

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

    return supported_files
