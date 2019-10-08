# I3D Action recognition

This is an implementation of the I3D action recognition model.

Due to difficulties linked to the nature of the dataset as youtube videos, results were computed using a roughly 90% subset of the original validation sets. As such the results cannot be compared directly to the ones reported from the original papers, although we should expect them to be close. All results are single-crop accuracy of the models with weights converted from the original [Tensorflow implementation](https://github.com/deepmind/kinetics-i3d).

## Kinetics 400
| Pre-training dataset | Input modality | Top 1 | Original Top 1 |
|:-:|:-:|:-:|:-:|
| None | RGB | 65.8 | 68.4 |
| None | Flow | 58.9 | 61.5 |
| None | RGB + Flow | 69.5 | 71.6 |
| ImageNet | RGB | 69.8 | 71.1 |
| ImageNet | Flow | 62.2 | 63.4 |
| ImageNet | RGB + Flow | 73.0 | 74.2 |

Original figures from ["Quo Vadis, Action Recognition? A New Model and the Kinetics Dataset"](https://arxiv.org/abs/1705.07750).

## Kinetics 600
| Pre-training dataset | Input modality | Top 1 | Original Top 1 |
|:-:|:-:|:-:|:-:|
| None | RGB | 68.4 | 71.9 |

Original figures from ["A Short Note about Kinetics-600"](https://arxiv.org/abs/1808.01340).

## Evaluation
To perform evaluation, you need to first extract the RGB frames from the original dataset, as well as the optical flow if you want to perform evaluation for a model requiring optical flow. This is important for performance reasons, as video decoding and especially optical flow computation are extremely time-consuming.

Assuming the Kinetics dataset has been downloaded in the format supported by `chainercv.datasets.kinetics.kinetics_dataset.KineticsDataset`, you can run the precomputation procedure as follows:

    python3 examples/i3d/precompute_frames_flow.py <KINETICS_VALIDATION_DIRECTORY> <OUTPUT_DIRECTORY>

You may disable optical flow computation by adding the flag `--no_compute_flow`, which will speed up this script considerably. Note that this script is parallelizable over multiple nodes using ChainerMN, and multiple processes on one node. The number of processes per node may be adjusted using the `--num_preprocess_workers` flag. This will require roughly 200GB of free space for the Kinetics-600 validation set without optical flow, and roughly 180GB of free space for the Kinetics-400 validation set with optical flow. Depending on your configuration, it may take up to many hours for RGB only, or up to multiple days if optical flow computation is required.

Once this process is finished, you may evaluate the models by running the following:

    python3 examples/i3d/eval_kinetics.py <PRECOMPUTED_KINETICS_DIRECTORY> \
        --rgb_model_checkpoint <CHECKPOINT_NAME> \
        --flow_model_checkpoint <CHECKPOINT_NAME>
 
At least one of `--rgb_model_checkpoint` or `--flow_model_checkpoint` should be specified. Both parameters accept either a path to a `.npz` checkpoint file, or one of the following:
- RGB models:
    - `rgb_scratch_kinetics600`
    - `rgb_scratch_kinetics400`
    - `rgb_imagenet_kinetics400`
- Flow models:
    - `flow_scratch_kinetics400`
    - `flow_imagenet_kinetics400`

This script is parallelizable over multiple GPUs and multiple nodes using ChainerMN. Additionally, you can specify the per-worker evaluation batch size, number of data loading processes using the `--batch_size` and `--num_preprocess_workers` flags. If you have installed [chainerio](https://github.com/chainer/chainerio), you may use the `--use_chainerio` flag which will allow loading data from HDFS. Using 64 P100 12GB GPUs, this process takes about 10 to 20 minutes to compute. The results will be printed to the console.

## Converting checkpoints from Tensorflow
The original Tensorflow checkpoint files (i.e. `xxx/model.ckpt` ) as available from [the original implementation](https://github.com/deepmind/kinetics-i3d) may be converted to Chainer's `.npz` format using the following script:

    python3 examples/i3d/tfckpt2npz.py <TENSORFLOW_CHECKPOINT> <OUTPUT_NPZ_CHECKPOINT>
 