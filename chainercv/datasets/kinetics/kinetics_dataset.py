import glob
import os
from multiprocessing import Pool, cpu_count
from warnings import warn

import numpy as np

from chainercv.chainer_experimental.datasets.sliceable import GetterDataset
from chainercv.utils.video.read_video import read_video


def _check_video_path(video_path):
    try:
        read_video(video_path, start_frame=0, end_frame=1)
    except IOError:
        return False
    except ValueError:
        return False
    return True


class KineticsDataset(GetterDataset):
    """Kinetics action recognition dataset.

    .. _ `Kinetics dataset`: https://deepmind.com/research/open-source/open-source-datasets/kinetics/

    .. note::
        Please manually download the dataset, as it is too large to be
        downloaded automatically.

    Args:
        data_dir (string): Path to the dataset directory. It should contain
        exactly one subdirectory for each label in the dataset. In turn, each
        label subdirectory should contain all the video samples for the
        dataset. Each file in the subdirectory should be readable by
        `chainercv.utils.read_video`.
        out_frames_per_second (int or None): Framerate in frames per second
        the videos should be produced at. If None, the original video
        framerate is used.

    This dataset returns the following data.

    .. csv-table::
        :header: name, shape, dtype, format

        :obj: `video`, ":math:`(T, 3, H, W)`", :obj: `float32`, \
        "RGB, :math:`[0, 255]`"
        :obj: `label`, ":math:`()`", :obj:`int32`, ":math:`[0, \#class - 1]`"
    """
    def __init__(self, data_dir, return_framerate=False,
                 num_video_check_workers=None, no_check_videos=False):
        super(KineticsDataset, self).__init__()
        class_dirs = glob.glob(os.path.join(data_dir, '*/'))

        if num_video_check_workers is None:
            num_video_check_workers = cpu_count()

        self.video_paths = []
        video_labels = []

        for class_dir in class_dirs:
            class_name = os.path.basename(os.path.normpath(class_dir))
            for video_path in glob.glob(os.path.join(class_dir, '*')):
                self.video_paths.append(video_path)
                video_labels.append(class_name)

        if not no_check_videos:
            # Filtering video which cannot be loaded. This may happen due to
            # corruption, unavailable videos from the original URL, or other
            # issues depending on the way the videos were downloaded. Since
            # this step can be expensive, it is parallelized over multiple
            # processes and made optional.
            pool = Pool(num_video_check_workers)

            video_is_available = pool.map(_check_video_path, self.video_paths)
            unavailable_videos = [path for i, path in
                                  enumerate(self.video_paths)
                                  if not video_is_available[i]]
            if len(unavailable_videos) > 0:
                warn("The following {} videos could not be loaded and have "
                     "been skipped: {}".format(len(unavailable_videos),
                                               "\n".join(unavailable_videos)))

            self.video_paths = [path for i, path in enumerate(self.video_paths)
                                if video_is_available[i]]
            video_labels = [label for i, label in enumerate(video_labels)
                            if video_is_available[i]]

        label_to_int = {l: i for i, l in enumerate(sorted(set(video_labels)))}
        self.video_labels = [
            label_to_int[l] for l in video_labels]
        self.return_framerate = return_framerate

        self.add_getter('video', self.get_video)
        self.add_getter('label', self.get_label)

    def get_video(self, i):
        video_path = self.video_paths[i]
        video = read_video(video_path, dtype=np.float32,
                           return_framerate=self.return_framerate)
        if self.return_framerate:
            video, framerate = video
        video_arr = np.array(video, dtype=np.float32)

        if not self.return_framerate:
            return video_arr
        else:
            return video_arr, framerate

    def get_label(self, i):
        return np.int32(self.video_labels[i])

    def __len__(self):
        return len(self.video_paths)
