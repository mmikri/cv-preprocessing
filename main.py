# main.py
import os
import argparse
from dotenv import load_dotenv
from app import __version__
from utils.__env import *
from utils.conf import load_config
from utils.file import create_output_directory, get_supported_files
from utils.preprocess import preprocess_media

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
        help="Path to images or video files (default: './assets')"
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

def main(args: argparse.Namespace) -> None:
    print(f"==================================")
    print(f"The app version is {__version__}")
    print(f"==================================")

    config_data = load_config(args.config)
    output_directory = args.output
    create_output_directory(output_directory)
    data_type = args.type
    input_path = args.path

    supported_files = get_supported_files(input_path, data_type)

    if supported_files:
        media_settings = config_data.get("image_settings", {}) if data_type in ["image", "i"] else config_data.get("video_settings", {})
        preprocess_media(supported_files, output_directory, media_settings)

if __name__ == "__main__":
    command_args = create_parser().parse_args()
    main(command_args)
