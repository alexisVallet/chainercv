import argparse
import os
from itertools import islice, cycle
import glob

import numpy as np
import chainer
import chainer.links as L
import chainermn

from chainercv.links.model.i3d.i3d import I3D
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


class I3DTransform(object):
    def __init__(self, resize_dim=256, crop_size=224,
                 sample_video_frames=79):
        self.resize_dim = resize_dim
        self.crop_size = crop_size
        self.sample_video_frames = sample_video_frames

    def __call__(self, inputs):
        video_arrays, label = inputs
        cropped_videos = []

        for video_array in video_arrays:
            # Only look at the first sample_video_frames frames. If not enough,
            # repeat the video.
            indices = list(islice(cycle(range(len(video_array))),
                                  self.sample_video_frames))
            video_array = video_array[indices]

            # Extract the center crop.
            h, w = video_array.shape[2:4]
            _, crop_params = center_crop(
                np.zeros((3, h, w), dtype=np.uint8),
                         (self.crop_size, self.crop_size), return_param=True)
            cropped_video = \
                video_array[:, :, crop_params['y_slice'],
                                crop_params['x_slice']]
            # NCHW to CNHW
            cropped_video = np.moveaxis(cropped_video, 0, 1)

            # Rescaling between -1 and 1
            cropped_video = (cropped_video.astype(np.float32) - 128) / 255
            cropped_videos.append(cropped_video)

        return tuple(cropped_videos + [label])


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

    for checkpoint, modality, num_classes in [_CHECKPOINTS[m] for m in args.models]:
        model = I3D(num_classes=num_classes, dropout_keep_prob=1.0)
        chainer.serializers.load_npz(checkpoint, model)
        models.append(model)

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
            sample_video_frames=args.sample_video_frames))

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
