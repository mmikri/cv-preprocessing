# cv-preprocessing

## Overview
This script performs preprocessing on image and video data based on specified configurations.

## Prerequisites
- Python 3.x

## Usage

### Installation
Clone the repository:
```bash
git clone https://github.com/mmikri/cv-preprocessing.git
cd cv-preprocessing
```

Install dependencies:
```bash
pip install -r requirements.txt
```

### Command Line Arguments
Element:
```bash
python3 tools.py [-h] [-t {image,i,video,v}] [-p PATH] [-c CONFIG] [-o OUTPUT]
```
- `-t, --type`: Specify the data type: 'image(i)' or 'video(v)' (default: 'image')
- `-p, --path`: Path to images or video files (default: './assets/teddybear.jpeg')
- `-c, --config`: Path to the configuration file (default: './config.yaml')
- `-o, --output`: Path to the output directory (default: './result')

Examples:
```bash
# Example 1: Process images
python preprocessing_script.py -t image -p ./path/to/images -c config.yaml -o ./output_directory

# Example 2: Process videos
python preprocessing_script.py -t video -p ./path/to/videos -c config.yaml -o ./output_directory
```

### Configuration File
The configuration file (`config.yaml`) contains settings for image and video preprocessing.
Customize this file based on your requirements.

```yaml
# Example Image Settings
image_settings:
  cropped:
    enabled: true
    coordinates: [10, 20, 200, 180]
  resize:
    enabled: true
    output_size: [300, 200]
  brightness:
    enabled: true
    factor: 1.5
  saturation:
    enabled: true
    factor: 0.8
  rotate:
    enabled: true
    angle: 1
  flip:
    enabled: true
    options: horizontally

# Example Video Settings
video_settings:
  cropped:
    enabled: true
    coordinates: [10, 20, 200, 180]
  resize:
    enabled: true
    output_size: [640, 480]
  brightness:
    enabled: true
    factor: 1.2
  saturation:
    enabled: true
    factor: 0.9
  rotate:
    enabled: true
    angle: 2
  flip:
    enabled: true
    options: vertically
  preview: true
  save_video: true
  save_image: true
```

### Run Unit Tests
To run unit tests, use the following command:
```bash
python -m unittest discover -s unittest -p 'test*.py'
```