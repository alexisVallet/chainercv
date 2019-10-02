"""
Precompute optical flow and individual frames from the Kinetics dataset.
This is useful for efficient evaluation and training of models when optical
flow computation or video decoding may otherwise create CPU bottlenecks.
"""
import argparse
import os
import zipfile

import cv2
import numpy as np
import PIL
import chainer
import chainermn

from chainercv.transforms.image.resize import resize
from chainercv.datasets.kinetics.kinetics_dataset import KineticsDataset


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
            "rgb": video_array,
            "label": label
        }

        if self.compute_flow:
            # Computing TVL1 optical flow
            optical_flow = cv2.DualTVL1OpticalFlow_create()
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

            out_dict["flow_x"] = chw_flow_video[:, 0]
            out_dict["flow_y"] = chw_flow_video[:, 1]
        return out_dict


class ZipSequenceEncodeTransform(object):
    def __init__(self, output_directory):
        self.output_directory = output_directory

    def __call__(self, sample):
        sample_id, sample = sample
        for modality in ("rgb", "flow_x", "flow_y"):
            if modality not in sample:
                continue
            video_array = sample[modality]
            out_dir = os.path.join(
                self.output_directory,
                modality,
                "class_{}".format(sample["label"])
            )
            os.makedirs(out_dir, exist_ok=True)
            sequence_filename = os.path.join(
                out_dir,
                "sample_{}.zip".format(sample_id)
            )
            with zipfile.ZipFile(sequence_filename, 'w') as sequence_zip:
                for i, frame in enumerate(video_array):
                    if modality == "rgb":
                        # Converting the input to HWC BGR so opencv is happy
                        frame_to_encode = cv2.cvtColor(
                            np.moveaxis(frame, 0, 2),
                            cv2.COLOR_RGB2BGR
                        )
                    elif modality.startswith("flow"):
                        # Already grayscale, just need to crop the values
                        # between -20 and 20, and bring it back to integer
                        # range 0-255.
                        frame_to_encode = (frame + 20) * 255 / 40
                        frame_to_encode = np.maximum(
                            0, np.minimum(255, np.round(frame_to_encode)))
                        frame_to_encode = frame_to_encode.astype(np.uint8)
                    _, jpeg_data = cv2.imencode('.jpg', frame_to_encode)
                    sequence_zip.writestr('{}.jpg'.format(i),
                                          jpeg_data.tobytes())

        return 0


class Enumerate(chainer.dataset.DatasetMixin):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def get_example(self, i):
        return i, self.dataset[i]


def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('dataset_directory')
    arg_parser.add_argument('output_directory')
    arg_parser.add_argument('--num_preprocess_workers', type=int, default=4)
    arg_parser.add_argument('--no_compute_flow', action='store_true')
    arg_parser.add_argument('--print_progress_num_batches', type=int,
                            default=1)
    args = arg_parser.parse_args()

    dataset = KineticsDataset(args.dataset_directory, return_framerate=True,
                              no_check_videos=True)
    # Split the dataset across workers.
    comm = chainermn.create_communicator()
    dataset = chainer.datasets.TransformDataset(dataset, VideoTransform())
    dataset = Enumerate(dataset)
    dataset = chainer.datasets.TransformDataset(
        dataset, ZipSequenceEncodeTransform(args.output_directory))
    dataset = chainermn.scatter_dataset(dataset, comm,
                                        force_equal_length=False)
    batch_size = args.num_preprocess_workers * 2
    iterator = chainer.iterators.MultiprocessIterator(
        dataset=dataset, batch_size=batch_size,
        n_processes=args.num_preprocess_workers, n_prefetch=2,
        repeat=False, shuffle=False)

    num_processed_videos = 0

    for i, b in enumerate(iterator):
        num_processed_videos += len(b)
        if i % args.print_progress_num_batches == 0 and comm.rank == 0:
            print("Processed {} out of {} videos for root worker...".format(
                num_processed_videos, len(dataset)))


if __name__ == '__main__':
    main()
