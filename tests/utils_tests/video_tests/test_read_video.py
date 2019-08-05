import unittest
import tempfile
import os

import numpy as np
try:
    import cv2  # NOQA
    _cv2_available = True
except ImportError:
    _cv2_available = False
from chainer import testing

from chainercv.utils.video.read_video import read_video


@testing.parameterize(*testing.product({
    'start_frame': (None, 4),
    'end_frame': (None, 12),
    'dtype': [np.float32, np.uint8]}))
class TestReadVideo(unittest.TestCase):
    def setUp(self):
        # Can't run tests without OpenCV.
        if not _cv2_available:
            return
        self.height = 32
        self.width = 64
        self.video = np.random.randint(
            0, 255, (16, 3, self.height, self.width), dtype=np.uint8)
        self.f = tempfile.NamedTemporaryFile(suffix='video.avi', delete=False)
        self.file = self.f.name
        writer = cv2.VideoWriter(
            self.file, cv2.VideoWriter_fourcc(*"MJPG"), 25,
            (self.width, self.height))
        for rgb_frame in self.video:
            bgr_frame = cv2.cvtColor(
                np.moveaxis(rgb_frame, 0, 2), cv2.COLOR_RGB2BGR)
            writer.write(bgr_frame)
        writer.release()

    def test_read_video(self):
        # Can't run tests without OpenCV.
        if not _cv2_available:
            return
        frame_iterator = read_video(
            file=self.file, start_frame=self.start_frame,
            end_frame=self.end_frame, dtype=self.dtype)
        frame_array = np.array(frame_iterator)
        if self.start_frame is None and self.end_frame is None:
            expected_length = self.video.shape[0]
        elif self.start_frame is None:
            expected_length = self.end_frame
        elif self.end_frame is None:
            expected_length = self.video.shape[0] - self.start_frame
        else:
            expected_length = self.end_frame - self.start_frame
        self.assertEqual(frame_array.shape,
                         (expected_length, 3, self.height, self.width))
        self.assertEqual(frame_array.dtype, self.dtype)

    def tearDown(self):
        os.remove(self.file)
