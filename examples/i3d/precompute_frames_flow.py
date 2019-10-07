"""
Precompute optical flow and individual frames from the Kinetics dataset.
This is useful for efficient evaluation and training of models when optical
flow computation or video decoding may otherwise create CPU bottlenecks.
"""
import argparse
import os
import zipfile
from itertools import count

import cv2
import numpy as np
import PIL
import chainer
import chainermn

from chainercv.transforms.image.resize import resize
from chainercv.datasets.kinetics.kinetics_dataset import KineticsDataset
from examples.i3d.image_sequence_dataset import ImageSequenceShardDataset


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


class JpegEncodeTransform(object):
    def __call__(self, sample):
        out_sample = {"label": sample["label"]}
        for modality in ("rgb", "flow_x", "flow_y"):
            if modality not in sample:
                continue
            video_array = sample[modality]
            out_sample[modality] = []
            for frame in video_array:
                if modality == "rgb":
                    # Converting the input to HWC BGR so opencv is happy
                    frame_to_encode = cv2.cvtColor(
                        np.moveaxis(frame, 0, 2),
                        cv2.COLOR_RGB2BGR)
                elif modality.startswith("flow"):
                    # Already grayscale, just need to crop the values
                    # between -20 and 20, and bring it back to integer
                    # range 0-255.
                    frame_to_encode = (frame + 20) * 255 / 40
                    frame_to_encode = np.maximum(
                        0, np.minimum(255, np.round(frame_to_encode)))
                    frame_to_encode = frame_to_encode.astype(np.uint8)
                _, jpeg_data = cv2.imencode('.jpg', frame_to_encode)
                out_sample[modality].append(jpeg_data)

        return out_sample


def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('dataset_directory')
    arg_parser.add_argument('output_directory')
    arg_parser.add_argument('--num_preprocess_workers', type=int, default=4)
    arg_parser.add_argument('--no_compute_flow', action='store_true')
    arg_parser.add_argument('--print_progress_num_shards', type=int,
                            default=1)
    arg_parser.add_argument('--num_samples_per_shard', type=int, default=128)
    arg_parser.add_argument('--skip_video_check', action='store_true',
                            help="If set, this flag skips a first pass over "
                                 "the dataset to check for corruption of "
                                 "the video files.")
    args = arg_parser.parse_args()

    dataset = KineticsDataset(args.dataset_directory, return_framerate=True,
                              no_check_videos=args.skip_video_check)
    # Split the dataset across workers.
    comm = chainermn.create_communicator()
    dataset = chainer.datasets.TransformDataset(dataset, VideoTransform(
        compute_flow=not args.no_compute_flow))
    dataset = chainer.datasets.TransformDataset(
        dataset, JpegEncodeTransform())

    # In order to make the precomputing restartable in case of failure of
    # some kind, fixing the seed for scattering across workers.
    dataset = chainermn.scatter_dataset(
        dataset, comm, force_equal_length=False, shuffle=True,
        seed=394727)

    # Shuffling the samples is desirable for training purposes, and of no
    # adverse consequence during evaluation, so performing it by default.
    # For this part, fixing the seed to be different for each worker so that
    # each worker shuffles its own shard differently. Datasets are shuffled
    # rather than iterators so that we can properly resume in the middle.
    shard_order = np.random.RandomState(48987 + comm.rank).permutation(
        len(dataset))
    dataset = chainer.datasets.SubDataset(
        dataset, start=0, finish=len(dataset), order=shard_order)

    # Checking for existing shards, if any.
    start_sample = 0
    start_shard = 0

    for i in count():
        start_shard = i
        shard_filename = os.path.join(
            args.output_directory, "shard_{}_{}.zip".format(comm.rank, i))
        try:
            shard_dataset = ImageSequenceShardDataset(shard_filename)
        except IOError:
            break
        start_sample += len(shard_dataset)
    if start_sample == len(dataset):
        print("Worker {}: all samples are already computed.".format(comm.rank))
        return
    if start_sample > 0:
        print("Worker {}: resuming from sample {}".format(
            comm.rank, start_sample))
    # Skipping already processed samples.
    dataset = chainer.datasets.SubDataset(
        dataset, start=start_sample, finish=len(dataset))

    iterator = chainer.iterators.MultiprocessIterator(
        dataset=dataset, batch_size=args.num_samples_per_shard,
        n_processes=args.num_preprocess_workers, n_prefetch=2,
        repeat=False, shuffle=False)

    num_processed_samples = start_sample
    os.makedirs(args.output_directory, exist_ok=True)
    tmp_dir = os.path.join(args.output_directory, 'tmp')
    os.makedirs(tmp_dir, exist_ok=True)

    for i, shard in zip(count(start_shard), iterator):
        # Writing the shard to a temporary file, then moving it once it is
        # finished writing to ensure atomic writes assuming a posix
        # filesystem.
        tmp_shard_filename = os.path.join(
            tmp_dir, "shard_{}_{}.zip".format(comm.rank, i))
        with zipfile.ZipFile(tmp_shard_filename, "w") as shard_zipfile:
            for sample_id, sample in enumerate(shard):
                sample_dir = "sample_{}".format(sample_id)
                shard_zipfile.writestr(
                    "{}/label".format(sample_dir), str(sample["label"]))

                for modality in ("rgb", "flow_x", "flow_y"):
                    if modality not in sample:
                        continue
                    video_array = sample[modality]
                    for frame_id, frame_jpeg_data in enumerate(video_array):
                        frame_key = "{}/{}/{}.jpg".format(
                            sample_dir, modality, frame_id)
                        shard_zipfile.writestr(frame_key,
                                               frame_jpeg_data.tobytes())
        shard_filename = os.path.join(
            args.output_directory, "shard_{}_{}.zip".format(comm.rank, i))
        os.rename(tmp_shard_filename, shard_filename)

        num_processed_samples += len(shard)

        if i % args.print_progress_num_shards == 0 and comm.rank == 0:
            print("Processed {} out of {} samples for root worker.".format(
                num_processed_samples, start_shard + len(dataset) - 1))


if __name__ == '__main__':
    main()
