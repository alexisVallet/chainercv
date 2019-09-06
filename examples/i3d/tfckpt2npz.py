import argparse

import tensorflow as tf
import numpy as np
import chainer

from chainercv.links.model.i3d.i3d import I3D


def load_tensorflow_checkpoint(i3d, checkpoint_filename,
                               ignore_batchnorm_statistics=False):
    tf_scope_to_unit3d = {
        "Conv3d_1a_7x7": i3d.conv3d_1a_7x7,
        "Conv3d_2b_1x1": i3d.conv3d_2b_1x1,
        "Conv3d_2c_3x3": i3d.conv3d_2c_3x3,
        "Logits/Conv3d_0c_1x1": i3d.logits_conv3d_0c_1x1,
        # Mixed 3b
        "Mixed_3b/Branch_0/Conv3d_0a_1x1": i3d.mixed_3b_branch_0_conv3d_0a_1x1,
        "Mixed_3b/Branch_1/Conv3d_0a_1x1": i3d.mixed_3b_branch_1_conv3d_0a_1x1,
        "Mixed_3b/Branch_1/Conv3d_0b_3x3": i3d.mixed_3b_branch_1_conv3d_0b_3x3,
        "Mixed_3b/Branch_2/Conv3d_0a_1x1": i3d.mixed_3b_branch_2_conv3d_0a_1x1,
        "Mixed_3b/Branch_2/Conv3d_0b_3x3": i3d.mixed_3b_branch_2_conv3d_0b_3x3,
        "Mixed_3b/Branch_3/Conv3d_0b_1x1": i3d.mixed_3b_branch_3_conv3d_0b_1x1,
        # Mixed 3c
        "Mixed_3c/Branch_0/Conv3d_0a_1x1": i3d.mixed_3c_branch_0_conv3d_0a_1x1,
        "Mixed_3c/Branch_1/Conv3d_0a_1x1": i3d.mixed_3c_branch_1_conv3d_0a_1x1,
        "Mixed_3c/Branch_1/Conv3d_0b_3x3": i3d.mixed_3c_branch_1_conv3d_0b_3x3,
        "Mixed_3c/Branch_2/Conv3d_0a_1x1": i3d.mixed_3c_branch_2_conv3d_0a_1x1,
        "Mixed_3c/Branch_2/Conv3d_0b_3x3": i3d.mixed_3c_branch_2_conv3d_0b_3x3,
        "Mixed_3c/Branch_3/Conv3d_0b_1x1": i3d.mixed_3c_branch_3_conv3d_0b_1x1,
        # Mixed 4b
        "Mixed_4b/Branch_0/Conv3d_0a_1x1": i3d.mixed_4b_branch_0_conv3d_0a_1x1,
        "Mixed_4b/Branch_1/Conv3d_0a_1x1": i3d.mixed_4b_branch_1_conv3d_0a_1x1,
        "Mixed_4b/Branch_1/Conv3d_0b_3x3": i3d.mixed_4b_branch_1_conv3d_0b_3x3,
        "Mixed_4b/Branch_2/Conv3d_0a_1x1": i3d.mixed_4b_branch_2_conv3d_0a_1x1,
        "Mixed_4b/Branch_2/Conv3d_0b_3x3": i3d.mixed_4b_branch_2_conv3d_0b_3x3,
        "Mixed_4b/Branch_3/Conv3d_0b_1x1": i3d.mixed_4b_branch_3_conv3d_0b_1x1,
        # Mixed 4c
        "Mixed_4c/Branch_0/Conv3d_0a_1x1": i3d.mixed_4c_branch_0_conv3d_0a_1x1,
        "Mixed_4c/Branch_1/Conv3d_0a_1x1": i3d.mixed_4c_branch_1_conv3d_0a_1x1,
        "Mixed_4c/Branch_1/Conv3d_0b_3x3": i3d.mixed_4c_branch_1_conv3d_0b_3x3,
        "Mixed_4c/Branch_2/Conv3d_0a_1x1": i3d.mixed_4c_branch_2_conv3d_0a_1x1,
        "Mixed_4c/Branch_2/Conv3d_0b_3x3": i3d.mixed_4c_branch_2_conv3d_0b_3x3,
        "Mixed_4c/Branch_3/Conv3d_0b_1x1": i3d.mixed_4c_branch_3_conv3d_0b_1x1,
        # Mixed 4d
        "Mixed_4d/Branch_0/Conv3d_0a_1x1": i3d.mixed_4d_branch_0_conv3d_0a_1x1,
        "Mixed_4d/Branch_1/Conv3d_0a_1x1": i3d.mixed_4d_branch_1_conv3d_0a_1x1,
        "Mixed_4d/Branch_1/Conv3d_0b_3x3": i3d.mixed_4d_branch_1_conv3d_0b_3x3,
        "Mixed_4d/Branch_2/Conv3d_0a_1x1": i3d.mixed_4d_branch_2_conv3d_0a_1x1,
        "Mixed_4d/Branch_2/Conv3d_0b_3x3": i3d.mixed_4d_branch_2_conv3d_0b_3x3,
        "Mixed_4d/Branch_3/Conv3d_0b_1x1": i3d.mixed_4d_branch_3_conv3d_0b_1x1,
        # Mixed 4e
        "Mixed_4e/Branch_0/Conv3d_0a_1x1": i3d.mixed_4e_branch_0_conv3d_0a_1x1,
        "Mixed_4e/Branch_1/Conv3d_0a_1x1": i3d.mixed_4e_branch_1_conv3d_0a_1x1,
        "Mixed_4e/Branch_1/Conv3d_0b_3x3": i3d.mixed_4e_branch_1_conv3d_0b_3x3,
        "Mixed_4e/Branch_2/Conv3d_0a_1x1": i3d.mixed_4e_branch_2_conv3d_0a_1x1,
        "Mixed_4e/Branch_2/Conv3d_0b_3x3": i3d.mixed_4e_branch_2_conv3d_0b_3x3,
        "Mixed_4e/Branch_3/Conv3d_0b_1x1": i3d.mixed_4e_branch_3_conv3d_0b_1x1,
        # Mixed 3b
        "Mixed_4f/Branch_0/Conv3d_0a_1x1": i3d.mixed_4f_branch_0_conv3d_0a_1x1,
        "Mixed_4f/Branch_1/Conv3d_0a_1x1": i3d.mixed_4f_branch_1_conv3d_0a_1x1,
        "Mixed_4f/Branch_1/Conv3d_0b_3x3": i3d.mixed_4f_branch_1_conv3d_0b_3x3,
        "Mixed_4f/Branch_2/Conv3d_0a_1x1": i3d.mixed_4f_branch_2_conv3d_0a_1x1,
        "Mixed_4f/Branch_2/Conv3d_0b_3x3": i3d.mixed_4f_branch_2_conv3d_0b_3x3,
        "Mixed_4f/Branch_3/Conv3d_0b_1x1": i3d.mixed_4f_branch_3_conv3d_0b_1x1,
        # Mixed 5b
        "Mixed_5b/Branch_0/Conv3d_0a_1x1": i3d.mixed_5b_branch_0_conv3d_0a_1x1,
        "Mixed_5b/Branch_1/Conv3d_0a_1x1": i3d.mixed_5b_branch_1_conv3d_0a_1x1,
        "Mixed_5b/Branch_1/Conv3d_0b_3x3": i3d.mixed_5b_branch_1_conv3d_0b_3x3,
        "Mixed_5b/Branch_2/Conv3d_0a_1x1": i3d.mixed_5b_branch_2_conv3d_0a_1x1,
        "Mixed_5b/Branch_2/Conv3d_0a_3x3": i3d.mixed_5b_branch_2_conv3d_0b_3x3,
        "Mixed_5b/Branch_3/Conv3d_0b_1x1": i3d.mixed_5b_branch_3_conv3d_0b_1x1,
        # Mixed 5c
        "Mixed_5c/Branch_0/Conv3d_0a_1x1": i3d.mixed_5c_branch_0_conv3d_0a_1x1,
        "Mixed_5c/Branch_1/Conv3d_0a_1x1": i3d.mixed_5c_branch_1_conv3d_0a_1x1,
        "Mixed_5c/Branch_1/Conv3d_0b_3x3": i3d.mixed_5c_branch_1_conv3d_0b_3x3,
        "Mixed_5c/Branch_2/Conv3d_0a_1x1": i3d.mixed_5c_branch_2_conv3d_0a_1x1,
        "Mixed_5c/Branch_2/Conv3d_0b_3x3": i3d.mixed_5c_branch_2_conv3d_0b_3x3,
        "Mixed_5c/Branch_3/Conv3d_0b_1x1": i3d.mixed_5c_branch_3_conv3d_0b_1x1,
    }
    ckpt_reader = tf.train.NewCheckpointReader(checkpoint_filename)
    # Some checkpoint have an additional prepended RGB or Flow scope.
    first_scope = list(ckpt_reader.get_variable_to_shape_map().keys())[0]
    allowed_scopes = ['RGB/inception_i3d/', 'Flow/inception_i3d/']
    prefix = ''
    for scope in allowed_scopes:
        if first_scope.startswith(scope):
            prefix = scope

    for tf_scope, unit3d in tf_scope_to_unit3d.items():
        tf_scope = prefix + tf_scope
        # Converting weights from Tensorflow's [t, h, w, c_i, c_o] layout to
        # Chainer's [c_o, c_i, t, h, w] layout.
        weights = ckpt_reader.get_tensor(
            tf_scope + '/conv_3d/w'
        )
        beta = None
        bias = None
        moving_mean = None
        moving_variance = None
        # The batch normalization and bias parameters (if any) are flattened.
        if unit3d.use_bias:
            bias = ckpt_reader.get_tensor(
                tf_scope + '/conv_3d/b'
            )

        if unit3d.use_batch_norm:
            beta = ckpt_reader.get_tensor(
                tf_scope + '/batch_norm/beta'
            )
            if not ignore_batchnorm_statistics:
                moving_mean = ckpt_reader.get_tensor(
                    tf_scope + '/batch_norm/moving_mean'
                )
                moving_variance = ckpt_reader.get_tensor(
                    tf_scope + '/batch_norm/moving_variance'
                )
        unit3d_load_tensorflow_weights(unit3d, weights=weights, bias=bias,
                                       beta=beta,
                                       moving_mean=moving_mean,
                                       moving_variance=moving_variance)


def unit3d_load_tensorflow_weights(unit3d, weights, bias=None, beta=None,
                                   moving_mean=None, moving_variance=None):
        weights = np.moveaxis(weights, -1, 0)
        weights = np.moveaxis(weights, -1, 1)
        unit3d.conv3d.conv.W.initializer = chainer.initializers.Constant(weights)
        if unit3d.use_bias:
            assert bias is not None, "Bias not provided even though use_bias has been set to True."
            bias = bias.flatten()
            unit3d.conv3d.conv.b.initializer = chainer.initializers.Constant(bias)
            unit3d.conv3d.conv.b.initialize(shape=bias.shape)
        if unit3d.use_batch_norm:
            assert beta is not None, (
                "Batch normalization parameters not provided even though use_batch_norm has been set to True."
            )
            beta = beta.flatten()
            beta_initializer = chainer.initializers.Constant(beta)
            beta_initializer.dtype = unit3d.bn._highprec_dtype
            unit3d.bn.beta = chainer.Parameter(beta_initializer)
            if moving_mean is not None:
                moving_mean = moving_mean.flatten()
                unit3d.bn._initial_avg_mean = moving_mean
            if moving_variance is not None:
                moving_variance = moving_variance.flatten()
                unit3d.bn._initial_avg_var = moving_variance
            unit3d.bn._initialize_params(beta.shape[0])


def tf_checkpoint_to_npz(tensorflow_model_checkpoint, output_npz_checkpoint, num_channels,
                         num_classes):
    model = I3D(num_classes=num_classes, dropout_keep_prob=1.0)
    load_tensorflow_checkpoint(model, tensorflow_model_checkpoint)
    # Need to input a dummy batch to force proper initialization of the weights.
    with chainer.using_config('train', False):
        dummy = np.zeros((2, num_channels, 16, 224, 224), dtype=np.float32)
        _ = model(dummy)
    chainer.serializers.save_npz(
        output_npz_checkpoint, model)


def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('tensorflow_model_checkpoint',
                            help="Path to the tensorflow model checkpoint "
                                 "to convert to npz.")
    arg_parser.add_argument('output_npz_checkpoint',
                            help="Path to the output .npz model checkpoint "
                                 "to generate.")
    arg_parser.add_argument('num_channels', type=int, help="Number of input channels of the model. 3 for RGB, 2 "
                                                           "for optical flow.")
    arg_parser.add_argument('num_classes', type=int, help="Output number of classes. 400 or 600 for kinetics.")
    args = arg_parser.parse_args()
    tf_checkpoint_to_npz(args.tensorflow_model_checkpoint, args.output_npz_checkpoint, args.num_channels,
                         args.num_classes)


if __name__ == '__main__':
    main()
