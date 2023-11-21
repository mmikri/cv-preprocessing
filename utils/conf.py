# config_utils.py
import yaml
from loguru import logger
from typing import Dict
from utils.__env import *

def load_config(config_file_path: str) -> Dict:
    try:
        with open(config_file_path) as config_stream:
            return yaml.safe_load(config_stream)
    except Exception as e:
        logger.error(f"Error occurred while loading YAML: {e}")
        sys.exit(1)