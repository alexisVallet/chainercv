import glob
import os

import numpy as np

from chainercv.chainer_experimental.datasets.sliceable import GetterDataset
from chainercv.utils.video.read_video import read_video


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

    This dataset returns the following data.

    .. csv-table::
        :header: name, shape, dtype, format

        :obj: `video`, ":math:`(T, 3, H, W)`", :obj: `float32`, \
        "RGB, :math:`[0, 255]`"
        :obj: `label`, ":math:`()`", :obj:`int32`, ":math:`[0, \#class - 1]`"
    """
    def __init__(self, data_dir):
        super(KineticsDataset, self).__init__()
        class_dirs = glob.glob(os.path.join(data_dir, '*/'))
        self.video_paths = []
        video_labels = []
        for class_dir in class_dirs:
            class_name = os.path.basename(os.path.normpath(class_dir))
            for video_path in glob.glob(os.path.join(class_dir, '*')):
                self.video_paths.append(video_path)
                video_labels.append(class_name)
        label_to_int = {l: i for i, l in enumerate(sorted(set(video_labels)))}
        self.video_labels = [
            label_to_int[l] for l in video_labels]

        self.add_getter('video', self.get_video)
        self.add_getter('label', self.get_label)

    def get_video(self, i):
        video_path = self.video_paths[i]

        video = read_video(video_path, dtype=np.float32)

        return np.array(video, dtype=np.float32)

    def get_label(self, i):
        return np.int32(self.video_labels[i])

    def __len__(self):
        return len(self.video_paths)
