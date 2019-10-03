import argparse
import os
from itertools import islice, cycle

import numpy as np
import chainer
import chainer.functions as F
import chainer.links as L
import chainermn

from chainercv.links.model.i3d.i3d import I3D
from chainercv.transforms.image.center_crop import center_crop
from examples.i3d.image_sequence_dataset import ImageSequenceDataset

_FLOW_CHECKPOINTS = {

    "flow_imagenet_kinetics400": (os.path.join('models', 'flow_imagenet_kinetics400.npz'), 400),
    "flow_scratch_kinetics400": (os.path.join('models', 'flow_scratch_kinetics400.npz'), 400)
}

_RGB_CHECKPOINTS = {
    "rgb_scratch_kinetics600": (os.path.join('models', 'rgb_scratch_kinetics600.npz'), 600),
    "rgb_imagenet_kinetics400": (os.path.join('models', 'rgb_imagenet_kinetics400.npz'), 400),
    "rgb_scratch_kinetics400": (os.path.join('models', 'rgb_scratch_kinetics400.npz'), 400),
}


class JointI3D(chainer.Chain):
    def __init__(self, rgb_model, flow_model):
        super(JointI3D, self).__init__()

        with self.init_scope():
            self.rgb_model = rgb_model
            self.flow_model = flow_model

    def __call__(self, rgb, flow):
        logits1 = self.rgb_model(rgb)
        logits2 = self.flow_model(flow)

        return logits1 + logits2


class I3DTransform(object):
    def __init__(self, crop_size=224, sample_video_frames=251,
                 modalities=("rgb", "flow")):
        self.crop_size = crop_size
        self.sample_video_frames = sample_video_frames
        self.modalities = modalities

    def __call__(self, inputs):
        video_arrays, label = inputs
        cropped_videos = {}
        expected_inputs = []
        if "rgb" in self.modalities:
            expected_inputs.append("rgb")
        if "flow" in self.modalities:
            expected_inputs.extend(["flow_x", "flow_y"])

        for modality in expected_inputs:
            video_array = video_arrays[modality]
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
            cropped_videos[modality] = cropped_video

        out_sample = []

        if "rgb" in cropped_videos:
            out_sample.append(cropped_videos["rgb"])
        if "flow_x" in cropped_videos:
            assert "flow_y" in cropped_videos
            flow = np.concatenate(
                (cropped_videos["flow_x"], cropped_videos["flow_y"]), axis=0)
            out_sample.append(flow)

        out_sample.append(label)

        return tuple(out_sample)


class I3DEvaluator(chainermn.extensions.GenericMultiNodeEvaluator):
    def calc_local(self, *args):
        target = self._targets['main']
        _ = target(*args)
        logits = target.y
        labels = args[-1]

        return logits, labels

    def aggregate(self, results):
        logits, labels = zip(*results)
        logits = F.concat(logits, axis=0)
        labels = F.concat(labels, axis=0)

        target = self._targets['main']
        acc = target.accfun(logits, labels)

        return acc


def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('val_dataset_directory',
                            help="Path to the Kinetics dataset directory.")
    arg_parser.add_argument('--rgb_model_checkpoint', default=None,
                            help="Checkpoint of the RGB model if any, "
                                 "At least one of 'rgb_model_checkpoint' "
                                 "or 'flow_model_checpoint' should be "
                                 "specified.",
                            choices=list(_RGB_CHECKPOINTS))
    arg_parser.add_argument('--flow_model_checkpoint', default=None,
                            help="Checkpoint of the flow model if any, "
                                 "At least one of 'rgb_model_checkpoint' "
                                 "or 'flow_model_checpoint' should be "
                                 "specified.",
                            choices=list(_FLOW_CHECKPOINTS))
    arg_parser.add_argument("--batch_size", type=int, default=2)
    arg_parser.add_argument("--num_preprocess_workers", type=int, default=2)
    arg_parser.add_argument("--sample_video_frames", type=int, default=251)
    args = arg_parser.parse_args()

    comm = chainermn.create_communicator()
    device = comm.intra_rank
    chainer.cuda.get_device_from_id(device).use()

    models = []
    modalities = []

    if args.rgb_model_checkpoint is not None:
        checkpoint, num_classes = _RGB_CHECKPOINTS[args.rgb_model_checkpoint]
        model = I3D(num_classes=num_classes, dropout_keep_prob=1.0)
        chainer.serializers.load_npz(checkpoint, model)
        models.append(model)
        modalities.append("rgb")
    if args.flow_model_checkpoint is not None:
        checkpoint, num_classes = _FLOW_CHECKPOINTS[args.flow_model_checkpoint]
        model = I3D(num_classes=num_classes, dropout_keep_prob=1.0)
        chainer.serializers.load_npz(checkpoint, model)
        models.append(model)
        modalities.append("flow")

    if len(models) == 0:
        raise ValueError("At least one of 'rgb_model_checkpoint' or "
                         "'flow_model_checpoint' should be specified.")
    elif len(models) == 2:
        model = JointI3D(*models)
    else:
        model = models[0]
    model = L.Classifier(predictor=model)
    model.to_gpu(device)

    # Setting up the dataset.
    dataset = ImageSequenceDataset(args.val_dataset_directory)
    dataset = chainer.datasets.TransformDataset(
        dataset=dataset, transform=I3DTransform(
            sample_video_frames=args.sample_video_frames,
            modalities=modalities))
    dataset = chainermn.scatter_dataset(dataset, comm,
                                        force_equal_length=False)

    iterator = chainer.iterators.MultiprocessIterator(
        dataset, batch_size=args.batch_size,
        n_processes=args.num_preprocess_workers, n_prefetch=2,
        repeat=False, shuffle=False)
    evaluator = I3DEvaluator(
        iterator=iterator, target=model, device=device, comm=comm)
    acc = evaluator(None)
    print("Top-1 accuracy {}".format(acc))


if __name__ == '__main__':
    main()
