import os
import threading
import zipfile
from itertools import count

import PIL
import chainer
import cv2
import numpy as np
import six

from chainercv.transforms import resize

try:
    import chainerio
    _chainerio_is_available = True
except ImportError:
    _chainerio_is_available = False


class SafeZipDataset(chainer.dataset.DatasetMixin):
    # Simple dataset returning members of a zip file without any
    # preprocessing. Implementation based on
    # chainer.datasets.ZippedImageDataset, credits go to the original authors!
    def __init__(self, zipfilename):
        self._zipfilename = zipfilename
        self._zf = zipfile.ZipFile(zipfilename)
        self._zf_pid = os.getpid()
        self.paths = [x for x in self._zf.namelist() if not x.endswith('/')]
        self._lock = threading.Lock()

    def __len__(self):
        return len(self.paths)

    def __getstate__(self):
        d = self.__dict__.copy()
        d['_zf'] = None
        d['_lock'] = None
        return d

    def __setstate__(self, state):
        self.__dict__ = state
        self._lock = threading.Lock()

    def get_example(self, i_or_filename):
        if isinstance(i_or_filename, six.integer_types):
            zfn = self.paths[i_or_filename]
        else:
            zfn = i_or_filename

        with self._lock:
            if self._zf is None or self._zf_pid != os.getpid():
                self._zf_pid = os.getpid()
                self._zf = zipfile.ZipFile(self._zipfilename)
            sample_data = self._zf.read(zfn)
        return sample_data


class ChainerIOSafeZipDataset(chainer.dataset.DatasetMixin):
    # Identical to SafeZipDataset, using chainerio backend for HDFS
    # support.
    def __init__(self, zipfilename):
        if not _chainerio_is_available:
            raise ValueError("chainerio needs to be installed to use "
                             "ChainerIOSafeZipDataset.")
        # Reading everything into shared memory ahead of time. Also avoids
        # issues with multiprocessing, as the data can be shared as-is across
        # processes without copy
        self._zipfilename = zipfilename
        self._zf = chainerio.open_as_container(self._zipfilename)
        self._zf_pid = os.getpid()
        self.paths = list(self._zf.list(recursive=True))
        self._lock = threading.Lock()

    def __len__(self):
        return len(self.paths)

    def __getstate__(self):
        d = self.__dict__.copy()
        d["_zf"] = None
        d["_lock"] = None
        return d

    def __setstate__(self, state):
        self.__dict__ = state
        self._lock = threading.Lock()

    def get_example(self, i_or_filename):
        if isinstance(i_or_filename, six.integer_types):
            zfn = self.paths[i_or_filename]
        else:
            zfn = i_or_filename

        with self._lock:
            if self._zf is None or self._zf_pid != os.getpid():
                self._zf_pid = os.getpid()
                self._zf = chainerio.open_as_container(self._zipfilename)
            sample_data = self._zf.open(zfn, 'rb').read()
        return sample_data


class ImageSequenceShardDataset(chainer.dataset.DatasetMixin):
    def __init__(self, shard_filename, use_chainerio=False):
        if not use_chainerio:
            self.actual_dataset = SafeZipDataset(shard_filename)
        else:
            self.actual_dataset = ChainerIOSafeZipDataset(shard_filename)
        self.num_samples = 0

        for sample_id in count():
            sample_label_key = "sample_{}/label".format(sample_id)
            if not sample_label_key in self.actual_dataset.paths:
                break
            self.num_samples += 1

        if self.num_samples == 0:
            raise ValueError("Could not find any samples in {}!".format(
                shard_filename))

    def __len__(self):
        return self.num_samples

    def get_example(self, i):
        sample_key = "sample_{}".format(i)
        sample_label_key = "{}/label".format(sample_key)
        sample = {}

        for modality in ("rgb", "flow_x", "flow_y"):
            modality_key = "{}/{}".format(sample_key, modality)
            video_frames = []

            for frame_id in count():
                frame_key = "{}/{}.jpg".format(modality_key, frame_id)
                if frame_key not in self.actual_dataset.paths:
                    break
                frame_jpeg_data = \
                    np.frombuffer(self.actual_dataset[frame_key],
                                  dtype=np.uint8)
                if modality == "rgb":
                    # Jpeg decode, bgr to rgb, hwc to chw
                    frame = cv2.imdecode(frame_jpeg_data, cv2.IMREAD_COLOR)
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame = np.moveaxis(frame, 2, 0)
                elif modality.startswith("flow"):
                    # Jpeg decode, adding channel dimension.
                    frame = cv2.imdecode(frame_jpeg_data,
                                         cv2.IMREAD_GRAYSCALE)
                    frame = frame[None, :, :]
                video_frames.append(frame)

            # Check whether the modality is available at all.
            if len(video_frames) == 0:
                continue
            sample[modality] = np.stack(video_frames, axis=0)

        label = int(self.actual_dataset[sample_label_key])

        return sample, label


class ImageSequenceDataset(chainer.datasets.ConcatenatedDataset):
    def __init__(self, shard_filenames, use_chainerio=False):
        datasets = [
            ImageSequenceShardDataset(f, use_chainerio=use_chainerio)
            for f in shard_filenames]
        super(ImageSequenceDataset, self).__init__(*datasets)


def resize_video(processed_video, resize_dim):
    # Resize so the smallest dimension is fixed.
    h, w = processed_video.shape[2:4]
    if h < w:
        new_size = (resize_dim, int(round(w * resize_dim / h)))
    else:
        new_size = (int(round(h * resize_dim / w)), resize_dim)
    resized_video = []
    num_frames = processed_video.shape[0]
    for i in range(num_frames):
        resized_video.append(resize(processed_video[i], new_size,
                                    interpolation=PIL.Image.BILINEAR))
    resized_video = np.stack(resized_video, axis=0)

    return resized_video


class VideoTransform(object):
    def __init__(self, resize_dim=256, compute_flow=True):
        self.resize_dim = resize_dim
        self.compute_flow = compute_flow

    def __call__(self, inputs):
        (video_array, orig_framerate), label = inputs

        # Resample the video to 25 frames per second
        total_video_length = len(video_array) / orig_framerate
        sample_frame_indices = np.linspace(
            0, len(video_array) - 1,
            max(1, total_video_length * 25), dtype=np.int64)
        video_array = video_array[sample_frame_indices]

        video_array = resize_video(video_array, self.resize_dim)

        out_dict = {
            "rgb": video_array
        }

        if self.compute_flow:
            # Computing TVL1 optical flow
            optical_flow = cv2.optflow.DualTVL1OpticalFlow_create()
            flow_array = []

            num_frames = video_array.shape[0]
            hwc_video = np.moveaxis(video_array, 1, 3).astype(np.uint8)
            prev_grayscale_frame = None

            for i in range(num_frames):
                rgb_frame = hwc_video[i]
                grayscale_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2GRAY)
                if prev_grayscale_frame is not None:
                    flow_array.append(optical_flow.calc(prev_grayscale_frame,
                                                        grayscale_frame, None))
                prev_grayscale_frame = grayscale_frame

            hwc_flow_video = np.stack(flow_array, axis=0)
            chw_flow_video = np.moveaxis(hwc_flow_video, 3, 1)
            # Bring flow to 0-255 integer range, truncating OF between
            # -20 and 20.
            chw_flow_video = (chw_flow_video + 20) * 255 / 40
            chw_flow_video = np.maximum(
                0, np.minimum(255, np.round(chw_flow_video)))
            chw_flow_video.astype(np.uint8)

            out_dict["flow_x"] = chw_flow_video[:, 0]
            out_dict["flow_y"] = chw_flow_video[:, 1]

        return out_dict, label
