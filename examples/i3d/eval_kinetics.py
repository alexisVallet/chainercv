import argparse
import os
from itertools import islice, cycle

import cv2
import numpy as np
import PIL
import chainer
import chainer.links as L
import chainermn

from chainercv.links.model.i3d.i3d import I3D
from chainercv.transforms.image.resize import resize
from chainercv.transforms.image.center_crop import center_crop
from chainercv.datasets.kinetics.kinetics_dataset import KineticsDataset

_CHECKPOINTS = {
    "rgb_scratch_kinetics600": (os.path.join('models', 'rgb_scratch_kinetics600.npz'), "rgb", 600),
    "rgb_imagenet_kinetics400": (os.path.join('models', 'rgb_imagenet_kinetics400.npz'), "rgb", 400),
    "flow_imagenet_kinetics400": (os.path.join('models', 'flow_imagenet_kinetics400.npz'), "flow", 400),
    "rgb_scratch_kinetics400": (os.path.join('models', 'rgb_scratch_kinetics400.npz'), "rgb", 400),
    "flow_scratch_kinetics400": (os.path.join('models', 'flow_scratch_kinetics400.npz'), "flow", 400)
}


class JointI3D(chainer.Chain):
    def __init__(self, model1, model2):
        super(JointI3D, self).__init__()

        with self.init_scope():
            self.model1 = model1
            self.model2 = model2

    def __call__(self, input1, input2):
        logits1 = self.model1(input1)
        logits2 = self.model2(input2)

        return logits1 + logits2


class VideoTransform(object):
    def __init__(self, modality):
        assert modality in ('rgb', 'flow')
        self.modality = modality

    def __call__(self, video_array):
        if self.modality == "rgb":
            # Linear scaling between -1 and 1
            return video_array / 128 - 1.
        elif self.modality == "flow":
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
                    flow_array.append(optical_flow.calc(prev_grayscale_frame, grayscale_frame, None))
                prev_grayscale_frame = grayscale_frame

            hwc_flow_video = np.stack(flow_array, axis=0)
            chw_flow_video = np.moveaxis(hwc_flow_video, 3, 1)

            chw_flow_video = np.maximum(-20, np.minimum(20, chw_flow_video))
            chw_flow_video /= 20.

            return chw_flow_video


class I3DTransform(object):
    def __init__(self, transforms, resize_dim=256, crop_size=224,
                 sample_video_frames=79):
        self.transforms = transforms
        self.resize_dim = resize_dim
        self.crop_size = crop_size
        self.sample_video_frames = sample_video_frames

    def __call__(self, inputs):
        (video_array, orig_framerate), label = inputs

        # Resample the video to 25 frames per second
        total_video_length = len(video_array) / orig_framerate
        sample_frame_indices = np.linspace(
            0, len(video_array) - 1,
            max(1, total_video_length * 25), dtype=np.int64)
        video_array = video_array[sample_frame_indices]

        # Only look at the first sample_video_frames frames. If not enough,
        # repeat the video.
        indices = list(islice(cycle(range(len(video_array))),
                              self.sample_video_frames))
        video_array = video_array[indices]

        cropped_videos = []

        for t in self.transforms:
            processed_video = t(video_array)

            # Resize so the smallest dimension is fixed.
            h, w = processed_video.shape[2:4]
            if h < w:
                new_size = (self.resize_dim, int(round(w * self.resize_dim / h)))
            else:
                new_size = (int(round(h * self.resize_dim / w)), self.resize_dim)
            resized_video = []
            num_frames = processed_video.shape[0]
            for i in range(num_frames):
                resized_video.append(resize(processed_video[i], new_size,
                                            interpolation=PIL.Image.BILINEAR))
            resized_video = np.stack(resized_video, axis=0)

            # Extract the center crop.
            _, crop_params = center_crop(np.zeros((3, new_size[0], new_size[1]), dtype=np.uint8),
                                         (self.crop_size, self.crop_size), return_param=True)
            cropped_video = resized_video[:, :, crop_params['y_slice'], crop_params['x_slice']]
            # NCHW to CNHW
            cropped_video = np.moveaxis(cropped_video, 0, 1)
            cropped_videos.append(cropped_video)

        return tuple(cropped_videos + [label])


def pad_concat(*args, **kwargs):
    return chainer.dataset.concat_examples(*args, padding=True, **kwargs)


def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('val_dataset_directory',
                            help="Path to the Kinetics dataset directory.")
    arg_parser.add_argument('models', choices=sorted(_CHECKPOINTS.keys()), nargs='*',
                            help="Model(s) to use jointly for prediction. Can be 1 or 2 models "
                                 "from the available choices. The chosen models should be trained "
                                 "on the same datasets and have the same number of classes for the "
                                 "results to be meaningful.")
    arg_parser.add_argument("--batch_size", type=int, default=2)
    arg_parser.add_argument("--num_preprocess_workers", type=int, default=2)
    arg_parser.add_argument("--sample_video_frames", type=int, default=251)
    args = arg_parser.parse_args()

    comm = chainermn.create_communicator()
    device = comm.intra_rank
    chainer.cuda.get_device_from_id(device).use()

    # Setting up the model
    if not 1 <= len(args.models) <= 2:
        raise ValueError("Either 1 or 2 models are required for evaluation!")

    if len(args.models) == 2:
        n_classes_1 = _CHECKPOINTS[args.models[0]][-1]
        n_classes_2 = _CHECKPOINTS[args.models[1]][-1]
        if not n_classes_1 == n_classes_2:
            raise ValueError("The 2 selected checkpoints have an incompatible number of classes: {}".format(
                ", ".join(args.models)))

    models = []
    transforms = []

    for checkpoint, modality, num_classes in [_CHECKPOINTS[m] for m in args.models]:
        model = I3D(num_classes=num_classes, dropout_keep_prob=1.0)
        chainer.serializers.load_npz(checkpoint, model)
        models.append(model)
        transforms.append(VideoTransform(modality=modality))

    if len(models) == 2:
        model = JointI3D(*models)
    else:
        model = models[0]
    model = L.Classifier(predictor=model)
    model.to_gpu(device)

    # Setting up the dataset.
    dataset = KineticsDataset(args.val_dataset_directory,
                              return_framerate=True)
    dataset = chainermn.scatter_dataset(dataset, comm, shuffle=True)
    dataset = chainer.datasets.TransformDataset(
        dataset=dataset, transform=I3DTransform(
            transforms, sample_video_frames=args.sample_video_frames))

    iterator = chainer.iterators.MultiprocessIterator(
        dataset, batch_size=args.batch_size,
        n_processes=args.num_preprocess_workers, n_prefetch=2,
        repeat=False, shuffle=False)
    evaluator = chainer.training.extensions.Evaluator(
        iterator=iterator, target=model, device=device, converter=pad_concat)
    evaluator = chainermn.create_multi_node_evaluator(
        actual_evaluator=evaluator, communicator=comm)
    results = evaluator()
    print(results)


if __name__ == '__main__':
    main()
