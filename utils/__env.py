import os
from dotenv import load_dotenv

load_dotenv()

VALID_IMAGE_TYPES = os.getenv("VALID_IMAGE_TYPES").split()
VALID_VIDEO_TYPES = os.getenv("VALID_VIDEO_TYPES").split()

IMAGE_EXT = os.getenv("IMAGE_EXT").split()
VIDEO_EXT = os.getenv("VIDEO_EXT").split()