import unittest
import os
import yaml
import cv2
import numpy as np
import shutil
import tempfile
from io import StringIO
from unittest.mock import patch
from tools import main, create_parser

class TestPreprocessingScript(unittest.TestCase):
    def setUp(self):
        # Set up temporary directory and file paths
        self.temp_dir = tempfile.mkdtemp(dir="")
        self.temp_image_path = os.path.join(self.temp_dir, 'temp_image.jpg')
        self.temp_video_path = os.path.join(self.temp_dir, 'temp_video.mp4')
        self.config_path = os.path.join(self.temp_dir, 'temp_config.yaml')
        print(f"\n")

    def tearDown(self):
        # Remove the used temporary directory
        shutil.rmtree(self.temp_dir)

    def create_temp_image(self, file_path):
        # Create a test image
        image = np.ones((100, 100, 3), dtype=np.uint8) * 255
        cv2.rectangle(image, (10, 10), (90, 90), (0, 0, 255), -1)
        cv2.imwrite(file_path, image)

    def create_temp_video(self, file_path):
        # Create a test video
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        frame_rate = 30
        video_writer = cv2.VideoWriter(file_path, fourcc, frame_rate, (100, 100))

        for _ in range(5):
            image = np.ones((100, 100, 3), dtype=np.uint8) * 255
            cv2.rectangle(image, (10, 10), (90, 90), (0, 0, 255), -1)
            video_writer.write(image)

        video_writer.release()

    def create_temp_config(self, config_path, config_data):
        # Create a YAML file for test configuration
        with open(config_path, 'w') as config_file:
            yaml.dump(config_data, config_file)

    def assertMediaProcessingCompleted(self, actual_output):
        # Check if the 'Media preprocessing completed' message is in the output string
        self.assertIn('Media preprocessing completed', actual_output)

    def assertMediaFileExists(self, file_path):
        # Check if a file exists at the given file path
        self.assertTrue(os.path.exists(file_path))

    def assertProcessedMediaMatchesOriginal(self, original_path, processed_path):
        # Check if the original and processed media match
        original_media = cv2.imread(original_path, cv2.IMREAD_UNCHANGED)
        processed_media = cv2.imread(processed_path, cv2.IMREAD_UNCHANGED)
        np.testing.assert_array_equal(original_media, processed_media)

    """
    Test1 : Media Processing with Able
    """
    def test_media_processing_with_able(self):
        # Test with virtual settings for images and videos
        test_cases = [
            ('image', self.temp_image_path),
            ('video', self.temp_video_path)
        ]

        # Virtual configuration data with all settings abled
        config_data = {
            'image_settings': {
                'cropped': {'enabled': True, 'coordinates': [10, 10, 90, 90]},
                'resize': {'enabled': True, 'output_size': [200, 200]},
                'brightness': {'enabled': True, 'factor': 0.6},
                'saturation': {'enabled': True, 'factor': 0.4},
                'rotate': {'enabled': True, 'angle': 1},
                'flip': {'enabled': True, 'options': 'horizontally'}
            },
            'video_settings': {
                'cropped': {'enabled': True, 'coordinates': [10, 10, 90, 90]},
                'resize': {'enabled': True, 'output_size': [200, 200]},
                'brightness': {'enabled': True, 'factor': 0.6},
                'saturation': {'enabled': True, 'factor': 0.4},
                'rotate': {'enabled': True, 'angle': 1},
                'flip': {'enabled': True, 'options': 'horizontally'},
                'preview': False,
                'save_video': True,
                'save_image': True
            }
        }

        for media_type, input_path in test_cases:
            with self.subTest(media_type=media_type):
                # Set up test data
                getattr(self, f'create_temp_{media_type}')(getattr(self, f'temp_{media_type}_path'))
                self.create_temp_config(self.config_path, config_data)

                # Execute the main function
                args = create_parser().parse_args(['-t', media_type, '-p', input_path, '-c', self.config_path, '-o', self.temp_dir])
                with patch('sys.stdout', new_callable=StringIO) as mock_stdout:
                    main(args)

                # Validation
                actual_output = mock_stdout.getvalue()
                self.assertMediaProcessingCompleted(actual_output)

                processed_media_path = os.path.join(self.temp_dir, os.path.basename(input_path))
                self.assertMediaFileExists(processed_media_path)
                self.assertProcessedMediaMatchesOriginal(input_path, processed_media_path)

    """
    Test2 : Media Processing with Disable
    """
    def test_media_processing_with_disable(self):
        # Test with virtual settings for images and videos
        test_cases = [
            ('image', self.temp_image_path),
            ('video', self.temp_video_path)
        ]

        # Virtual configuration data with all settings disabled
        config_data = {
            'image_settings': {
                'cropped': {'enabled': False},
                'resize': {'enabled': False},
                'brightness': {'enabled': False},
                'saturation': {'enabled': False},
                'rotate': {'enabled': False},
                'flip': {'enabled': False}
            },
            'video_settings': {
                'cropped': {'enabled': False},
                'resize': {'enabled': False},
                'brightness': {'enabled': False},
                'saturation': {'enabled': False},
                'rotate': {'enabled': False},
                'flip': {'enabled': False},
                'preview': False,
                'save_video': True,
                'save_image': True
            }
        }

        for media_type, input_path in test_cases:
            with self.subTest(media_type=media_type):
                # Set up test data
                getattr(self, f'create_temp_{media_type}')(getattr(self, f'temp_{media_type}_path'))
                self.create_temp_config(self.config_path, config_data)

                # Execute the main function
                args = create_parser().parse_args(['-t', media_type, '-p', input_path, '-c', self.config_path, '-o', self.temp_dir])
                with patch('sys.stdout', new_callable=StringIO) as mock_stdout:
                    main(args)

                # Validation
                actual_output = mock_stdout.getvalue()
                self.assertMediaProcessingCompleted(actual_output)

                processed_media_path = os.path.join(self.temp_dir, os.path.basename(input_path))
                self.assertMediaFileExists(processed_media_path)
                self.assertProcessedMediaMatchesOriginal(input_path, processed_media_path)

if __name__ == '__main__':
    unittest.main()