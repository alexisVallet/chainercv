import unittest
import tempfile
import os

import numpy as np
try:
    import cv2  # NOQA
    _cv2_available = True
except ImportError:
    _cv2_available = False

from chainercv.datasets.kinetics.kinetics_dataset import KineticsDataset


class TestKineticsDataset(unittest.TestCase):
    def setUp(self):
        if not _cv2_available:
            return
        self.kinetics_dir = tempfile.TemporaryDirectory()
        self.num_classes = 10
        self.num_samples_per_class = 10
        self.video_length = 16
        self.video_height = 32
        self.video_width = 64

        for i in range(self.num_classes):
            class_dir = os.path.join(self.kinetics_dir.name,
                                     'class_{}'.format(i))
            os.makedirs(class_dir)

            for j in range(self.num_samples_per_class):
                video = np.random.randint(
                    0, 255, size=(self.video_length, 3, self.video_height,
                                  self.video_width), dtype=np.uint8)
                writer = cv2.VideoWriter(
                    os.path.join(class_dir, 'sample_{}.avi'.format(j)),
                    cv2.VideoWriter_fourcc(*"MJPG"), 25, (self.video_width,
                                                          self.video_height))
                for rgb_frame in video:
                    bgr_frame = cv2.cvtColor(
                        np.moveaxis(rgb_frame, 0, 2), cv2.COLOR_RGB2BGR)
                    writer.write(bgr_frame)
                writer.release()

    def test_kinetics_dataset(self):
        if not _cv2_available:
            return

        dataset = KineticsDataset(self.kinetics_dir.name)
        self.assertEqual(len(dataset), self.num_classes *
                         self.num_samples_per_class)

        for i in range(len(dataset)):
            example = dataset[i]

            self.assertGreaterEqual(len(example), 2)

            video, label = example[:2]

            self.assertIsInstance(video, np.ndarray)
            self.assertEqual(
                video.shape, (self.video_length, 3, self.video_height,
                              self.video_width))
            self.assertEqual(video.dtype, np.float32)
            self.assertIsInstance(label, np.int32)
            self.assertGreaterEqual(label, 0)
            self.assertLess(label, self.num_classes)
