# I3D Action recognition
Due to difficulties linked to the nature of the dataset as youtube videos, results were computed using a roughly 90% of the original validation sets. All results are single-crop accuracy of the models with weights converted from the original Tensorflow implementation.


## Kinetics 400
| Pre-training dataset | Input modality | Top 1 | Original Top 1 |
|:-:|:-:|:-:|:-:|
| None | RGB + Flow | 69.5 | 71.6 |
| ImageNet | RGB + Flow | 73.0 | 74.2 |

## Kinetics 600
| Pre-training dataset | Input modality | Top 1 | Original Top 1 |
|:-:|:-:|:-:|:-:|
| None | RGB | 68.4 | 71.9 |

## Evaluation
To perform evaluation, you need to first extract the RGB frames from the original dataset, as well as the optical flow if you want to perform evaluation for a model requiring optical flow. This is important for performance reasons, as video decoding and especially optical flow computation are extremely time-consuming.

Assuming the Kinetics dataset has been downloaded in the format supported by `chainercv.datasets.kinetics.kinetics_dataset.KineticsDataset`, you can run the precomputation procedure as follows:

    python3 examples/i3d/precompute_frames_flow.py <KINETICS_VALIDATION_DIRECTORY> <OUTPUT_DIRECTORY>

You may disable optical flow computation by adding the flag `--no_compute_flow`, which will speed up this script considerably. Note that this script is parallelizable over multiple nodes using ChainerMN, and multiple processes on one node. The number of processes per node may be adjusted using the `--num_preprocess_workers` flag. This will require roughly 200GB of free space for the Kinetics-600 validation set without optical flow, and roughly 180GB of free space for the Kinetics-400 validation set with optical flow. It may take multiple hours for RGB only, multiple days if optical flow computation is required.

Once this process is finished, you may evaluate the models by running the following:

    python3 examples/i3d/eval_kinetics.py <PRECOMPUTED_KINETICS_DIRECTORY> \
        --rgb_model_checkpoint <CHECKPOINT_NAME> \
        --flow_model_checkpoint <CHECKPOINT_NAME>
 
This script is parallelizable over multiple GPUs and multiple nodes using ChainerMN. Additionally, you can specify the per-worker evaluation batch size, number of data loading processes using the `--batch_size` and `--num_preprocess_workers` flags. If you have installed [chainerio](https://github.com/chainer/chainerio), you may use the `--use_chainerio` to use it which will allow loading data from HDFS, for instance. Using 64 P100 12GB GPUs, this process takes about 10 to 20 minutes to compute. The results will be printed to the console.
