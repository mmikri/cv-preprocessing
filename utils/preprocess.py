# preprocess_utils.py
import os
import sys
from loguru import logger
from typing import List, Dict
from utils.__env import *
from utils.file import create_output_directory
from utils.cv import image_process, video_process

def preprocess_media(input_files: List[str], output_folder: str, settings: Dict) -> None:
    try:
        for input_file in input_files:
            sub_output_folder = os.path.abspath(os.path.join(output_folder, os.path.dirname(input_file)))
            create_output_directory(sub_output_folder)
            if input_file.lower().endswith(tuple(VIDEO_EXT)):
                video_process(input_file, sub_output_folder, settings)
            else:
                image_process(input_file, sub_output_folder, settings)
        print(f"==================================")
        print(f"Media preprocessing completed")
        print(f"==================================")
    except Exception as e:
        logger.error(f"Error occurred during media preprocessing: {e}")
        sys.exit(1)