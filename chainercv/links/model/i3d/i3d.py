"""
Implementation of the I3D model, using pre-trained weights from deepmind.
"""
import chainer
import chainer.functions as F
import chainer.links as L
from chainer.functions import pad
from chainer.links import Convolution3D
from chainer.utils import conv
import chainermn

import numpy as np


def _triplet(x):
    if hasattr(x, '__getitem__'):
        return x
    return x, x, x


def _get_pad(in_size, ksize, stride, tf_padding):
    if tf_padding == 'SAME':
        tf_out_size = int(np.ceil(float(in_size) / stride))
    elif tf_padding == 'VALID':
        tf_out_size = int(np.ceil(float(in_size - ksize + 1) / stride))
    pad = int(stride * tf_out_size - in_size + ksize - stride)
    assert conv.get_conv_outsize(in_size + pad, ksize, stride,
                                 0) == tf_out_size
    return pad


def _tf_padding(x, ksize, stride, tf_padding):
    pad_2 = _get_pad(x.shape[2], ksize[0], stride[0], tf_padding)
    pad_3 = _get_pad(x.shape[3], ksize[1], stride[1], tf_padding)
    pad_4 = _get_pad(x.shape[4], ksize[2], stride[2], tf_padding)
    if pad_2 or pad_3 or pad_4:
        return pad(
            x,
            ((0, 0), (0, 0),
             (pad_2 // 2, int(np.ceil(float(pad_2) / 2))),
             (pad_3 // 2, int(np.ceil(float(pad_3) / 2))),
             (pad_4 // 2, int(np.ceil(float(pad_4) / 2)))),
            mode='constant')
    else:
        return x


class TFConvolution3D(chainer.Chain):
    """Tensorflow compatible Convolution3D
    Based on okdshin's in progress (at the time of writing) Mobilenet V2
    PR to ChainerCV:
    https://github.com/chainer/chainercv/pull/838
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 ksize,
                 stride=1,
                 pad='SAME',
                 nobias=False,
                 initialW=None,
                 initial_bias=None,
                 **kwargs):
        super(TFConvolution3D, self).__init__()

        if pad in ('SAME', 'VALID'):  # TF compatible pad
            self.padding = lambda x: _tf_padding(x, _triplet(self.conv.ksize),
                                                 _triplet(self.conv.stride), pad)
        else:
            self.padding = None

        with self.init_scope():
            self.conv = Convolution3D(in_channels, out_channels, ksize, stride,
                                      0, nobias, initialW, initial_bias,
                                      **kwargs)

    @property
    def W(self):
        return self.conv.W

    @property
    def b(self):
        return self.conv.b

    def forward(self, x):
        if self.padding is None:
            return self.conv(x)
        else:
            return self.conv(self.padding(x))


class Unit3D(chainer.Chain):
    def __init__(self, output_channels, kernel_shape=(1, 1, 1),
                 stride=(1, 1, 1), activation_fn=F.relu, use_batch_norm=True,
                 use_bias=False, channel_factor=1, multi_node_bn_comm=None):
        super(Unit3D, self).__init__()
        self.activation_fn = activation_fn
        self.use_batch_norm = use_batch_norm
        self.use_bias = use_bias
        self.actual_output_channels = max(1, int(round(output_channels *
                                                       channel_factor)))
        with self.init_scope():
            self.conv3d = TFConvolution3D(
                in_channels=None,
                out_channels=self.actual_output_channels,
                ksize=kernel_shape,
                stride=stride,
                nobias=not use_bias,
            )
            if use_batch_norm:
                if multi_node_bn_comm is None:
                    self.bn = L.BatchNormalization(
                        size=self.actual_output_channels,
                        decay=0.999,
                        eps=1e-3,
                        use_gamma=False,
                    )
                else:
                    self.bn = chainermn.links.MultiNodeBatchNormalization(
                        size=self.actual_output_channels,
                        comm=multi_node_bn_comm,
                        decay=0.999,
                        eps=1e-3,
                        use_gamma=False,
                        dtype=chainer.config.dtype,
                    )

    def __call__(self, inputs):
        net = self.conv3d(inputs)
        if self.use_batch_norm:
            net = self.bn(net)
        if self.activation_fn is not None:
            net = self.activation_fn(net)
        return net


def max_pooling_3d_same(inputs, ksize, strides):
    padded_inputs = _tf_padding(inputs, ksize, strides, tf_padding='SAME')

    return F.max_pooling_3d(padded_inputs, ksize=ksize, stride=strides)


def maybe_add_features(layers, outputs, layer_name, x):
    if layer_name in layers:
        outputs[layer_name] = x


class I3D(chainer.Chain):
    # TODO: move those to the evaluation script.


    def __init__(self, num_classes, dropout_keep_prob,
                 time_strides=(2, 1, 1, 2, 2,),
                 time_ksizes=(7, 1, 1, 3, 2,), channel_factor=1,
                 multi_node_bn_comm=None, skip_logits=False):
        super(I3D, self).__init__()
        self.num_classes = num_classes
        self.dropout_keep_prob = dropout_keep_prob
        self.time_strides = time_strides
        self.time_ksizes = time_ksizes
        self.skip_logits = skip_logits

        with self.init_scope():
            # As in the Slowfast paper, setting the first layer with kernel
            # size 5 if stride is 1. Leaving as 7 as in the original I3D when
            # stride is 2. Although the Slowfast paper uses kernel size 1
            # for convolutions at the bottom, this is a performance
            # optimization which I leave out for now.
            self.conv3d_1a_7x7 = Unit3D(
                output_channels=64,
                kernel_shape=(time_ksizes[0], 7, 7),
                stride=(time_strides[0], 2, 2),
                channel_factor=channel_factor,
                multi_node_bn_comm=multi_node_bn_comm,
            )
            # Max pooling 1x3x3
            self.conv3d_2b_1x1 = Unit3D(
                output_channels=64,
                kernel_shape=(1, 1, 1),
                channel_factor=channel_factor,
                multi_node_bn_comm=multi_node_bn_comm,
            )
            self.conv3d_2c_3x3 = Unit3D(
                output_channels=192,
                kernel_shape=(3, 3, 3),
                channel_factor=channel_factor,
                multi_node_bn_comm=multi_node_bn_comm,
            )
            # Max pooling 1x3x3

            # Mixed 3b
            # Branch 0
            self.mixed_3b_branch_0_conv3d_0a_1x1 = Unit3D(
                output_channels=64, kernel_shape=(1, 1, 1),
                channel_factor=channel_factor,
                multi_node_bn_comm=multi_node_bn_comm, )
            # Branch 1
            self.mixed_3b_branch_1_conv3d_0a_1x1 = Unit3D(
                output_channels=96, kernel_shape=(1, 1, 1),
                channel_factor=channel_factor,
                multi_node_bn_comm=multi_node_bn_comm, )
            self.mixed_3b_branch_1_conv3d_0b_3x3 = Unit3D(
                output_channels=128, kernel_shape=(3, 3, 3),
                channel_factor=channel_factor,
                multi_node_bn_comm=multi_node_bn_comm, )
            # Branch 2
            self.mixed_3b_branch_2_conv3d_0a_1x1 = Unit3D(output_channels=16,
                                                          kernel_shape=(
                                                              1, 1, 1),
                                                          channel_factor=channel_factor,
                                                          multi_node_bn_comm=multi_node_bn_comm, )
            self.mixed_3b_branch_2_conv3d_0b_3x3 = Unit3D(output_channels=32,
                                                          kernel_shape=(
                                                              3, 3, 3),
                                                          channel_factor=channel_factor,
                                                          multi_node_bn_comm=multi_node_bn_comm, )
            # Branch 3
            # Max pooling 3x3x3 stride 1x1x1
            self.mixed_3b_branch_3_conv3d_0b_1x1 = Unit3D(output_channels=32,
                                                          kernel_shape=(
                                                              1, 1, 1),
                                                          channel_factor=channel_factor,
                                                          multi_node_bn_comm=multi_node_bn_comm, )
            # Concat branch 0 through 3

            # Mixed 3c
            # Branch 0
            self.mixed_3c_branch_0_conv3d_0a_1x1 = Unit3D(output_channels=128,
                                                          kernel_shape=(
                                                              1, 1, 1),
                                                          channel_factor=channel_factor,
                                                          multi_node_bn_comm=multi_node_bn_comm, )
            # Branch 1
            self.mixed_3c_branch_1_conv3d_0a_1x1 = Unit3D(output_channels=128,
                                                          kernel_shape=(
                                                              1, 1, 1),
                                                          channel_factor=channel_factor,
                                                          multi_node_bn_comm=multi_node_bn_comm, )
            self.mixed_3c_branch_1_conv3d_0b_3x3 = Unit3D(output_channels=192,
                                                          kernel_shape=(
                                                              3, 3, 3),
                                                          channel_factor=channel_factor,
                                                          multi_node_bn_comm=multi_node_bn_comm, )
            # Branch 2
            self.mixed_3c_branch_2_conv3d_0a_1x1 = Unit3D(output_channels=32,
                                                          kernel_shape=(
                                                              1, 1, 1),
                                                          channel_factor=channel_factor,
                                                          multi_node_bn_comm=multi_node_bn_comm, )
            self.mixed_3c_branch_2_conv3d_0b_3x3 = Unit3D(output_channels=96,
                                                          kernel_shape=(
                                                              3, 3, 3),
                                                          channel_factor=channel_factor,
                                                          multi_node_bn_comm=multi_node_bn_comm, )
            # Branch 3
            # Max pooling 3x3x3 stride 1x1x1
            self.mixed_3c_branch_3_conv3d_0b_1x1 = Unit3D(output_channels=64,
                                                          kernel_shape=(
                                                              1, 1, 1),
                                                          channel_factor=channel_factor,
                                                          multi_node_bn_comm=multi_node_bn_comm, )
            # Concat branch 0 through 3

            # Max pooling 3x3x3 stride 2x2x2

            # Mixed 4b
            # Branch 0
            self.mixed_4b_branch_0_conv3d_0a_1x1 = Unit3D(output_channels=192,
                                                          kernel_shape=(
                                                              1, 1, 1),
                                                          channel_factor=channel_factor,
                                                          multi_node_bn_comm=multi_node_bn_comm, )
            # Branch 1
            self.mixed_4b_branch_1_conv3d_0a_1x1 = Unit3D(output_channels=96,
                                                          kernel_shape=(
                                                              1, 1, 1),
                                                          channel_factor=channel_factor,
                                                          multi_node_bn_comm=multi_node_bn_comm, )
            self.mixed_4b_branch_1_conv3d_0b_3x3 = Unit3D(output_channels=208,
                                                          kernel_shape=(
                                                              3, 3, 3),
                                                          channel_factor=channel_factor,
                                                          multi_node_bn_comm=multi_node_bn_comm, )
            # Branch 2
            self.mixed_4b_branch_2_conv3d_0a_1x1 = Unit3D(output_channels=16,
                                                          kernel_shape=(
                                                              1, 1, 1),
                                                          channel_factor=channel_factor,
                                                          multi_node_bn_comm=multi_node_bn_comm, )
            self.mixed_4b_branch_2_conv3d_0b_3x3 = Unit3D(output_channels=48,
                                                          kernel_shape=(
                                                              3, 3, 3),
                                                          channel_factor=channel_factor,
                                                          multi_node_bn_comm=multi_node_bn_comm, )
            # Branch 3
            # Max pooling 3x3x3 stride 1x1x1
            self.mixed_4b_branch_3_conv3d_0b_1x1 = Unit3D(output_channels=64,
                                                          kernel_shape=(
                                                              1, 1, 1),
                                                          channel_factor=channel_factor,
                                                          multi_node_bn_comm=multi_node_bn_comm, )
            # Concat branch 0 through 3

            # Mixed 4c
            # Branch 0
            self.mixed_4c_branch_0_conv3d_0a_1x1 = Unit3D(output_channels=160,
                                                          kernel_shape=(
                                                              1, 1, 1),
                                                          channel_factor=channel_factor,
                                                          multi_node_bn_comm=multi_node_bn_comm, )
            # Branch 1
            self.mixed_4c_branch_1_conv3d_0a_1x1 = Unit3D(output_channels=112,
                                                          kernel_shape=(
                                                              1, 1, 1),
                                                          channel_factor=channel_factor,
                                                          multi_node_bn_comm=multi_node_bn_comm, )
            self.mixed_4c_branch_1_conv3d_0b_3x3 = Unit3D(output_channels=224,
                                                          kernel_shape=(
                                                              3, 3, 3),
                                                          channel_factor=channel_factor,
                                                          multi_node_bn_comm=multi_node_bn_comm, )
            # Branch 2
            self.mixed_4c_branch_2_conv3d_0a_1x1 = Unit3D(output_channels=24,
                                                          kernel_shape=(
                                                              1, 1, 1),
                                                          channel_factor=channel_factor,
                                                          multi_node_bn_comm=multi_node_bn_comm, )
            self.mixed_4c_branch_2_conv3d_0b_3x3 = Unit3D(output_channels=64,
                                                          kernel_shape=(
                                                              3, 3, 3),
                                                          channel_factor=channel_factor,
                                                          multi_node_bn_comm=multi_node_bn_comm, )
            # Branch 3
            # Max pooling 3x3x3 stride 1x1x1
            self.mixed_4c_branch_3_conv3d_0b_1x1 = Unit3D(output_channels=64,
                                                          kernel_shape=(
                                                              1, 1, 1),
                                                          channel_factor=channel_factor,
                                                          multi_node_bn_comm=multi_node_bn_comm, )
            # Concat branch 0 through 3

            # Mixed 4d
            # Branch 0
            self.mixed_4d_branch_0_conv3d_0a_1x1 = Unit3D(output_channels=128,
                                                          kernel_shape=(
                                                              1, 1, 1),
                                                          channel_factor=channel_factor,
                                                          multi_node_bn_comm=multi_node_bn_comm, )
            # Branch 1
            self.mixed_4d_branch_1_conv3d_0a_1x1 = Unit3D(output_channels=128,
                                                          kernel_shape=(
                                                              1, 1, 1),
                                                          channel_factor=channel_factor,
                                                          multi_node_bn_comm=multi_node_bn_comm, )
            self.mixed_4d_branch_1_conv3d_0b_3x3 = Unit3D(output_channels=256,
                                                          kernel_shape=(
                                                              3, 3, 3),
                                                          channel_factor=channel_factor,
                                                          multi_node_bn_comm=multi_node_bn_comm, )
            # Branch 2
            self.mixed_4d_branch_2_conv3d_0a_1x1 = Unit3D(output_channels=24,
                                                          kernel_shape=(
                                                              1, 1, 1),
                                                          channel_factor=channel_factor,
                                                          multi_node_bn_comm=multi_node_bn_comm, )
            self.mixed_4d_branch_2_conv3d_0b_3x3 = Unit3D(output_channels=64,
                                                          kernel_shape=(
                                                              3, 3, 3),
                                                          channel_factor=channel_factor,
                                                          multi_node_bn_comm=multi_node_bn_comm, )
            # Branch 3
            # Max pooling 3x3x3 stride 1x1x1
            self.mixed_4d_branch_3_conv3d_0b_1x1 = Unit3D(output_channels=64,
                                                          kernel_shape=(
                                                              1, 1, 1),
                                                          channel_factor=channel_factor,
                                                          multi_node_bn_comm=multi_node_bn_comm, )
            # Concat branch 0 through 3

            # Mixed 4e
            # Branch 0
            self.mixed_4e_branch_0_conv3d_0a_1x1 = Unit3D(output_channels=112,
                                                          kernel_shape=(
                                                              1, 1, 1),
                                                          channel_factor=channel_factor,
                                                          multi_node_bn_comm=multi_node_bn_comm, )
            # Branch 1
            self.mixed_4e_branch_1_conv3d_0a_1x1 = Unit3D(output_channels=144,
                                                          kernel_shape=(
                                                              1, 1, 1),
                                                          channel_factor=channel_factor,
                                                          multi_node_bn_comm=multi_node_bn_comm, )
            self.mixed_4e_branch_1_conv3d_0b_3x3 = Unit3D(output_channels=288,
                                                          kernel_shape=(
                                                              3, 3, 3),
                                                          channel_factor=channel_factor,
                                                          multi_node_bn_comm=multi_node_bn_comm, )
            # Branch 2
            self.mixed_4e_branch_2_conv3d_0a_1x1 = Unit3D(output_channels=32,
                                                          kernel_shape=(
                                                              1, 1, 1),
                                                          channel_factor=channel_factor,
                                                          multi_node_bn_comm=multi_node_bn_comm, )
            self.mixed_4e_branch_2_conv3d_0b_3x3 = Unit3D(output_channels=64,
                                                          kernel_shape=(
                                                              3, 3, 3),
                                                          channel_factor=channel_factor,
                                                          multi_node_bn_comm=multi_node_bn_comm, )
            # Branch 3
            # Max pooling 3x3x3 stride 1x1x1
            self.mixed_4e_branch_3_conv3d_0b_1x1 = Unit3D(output_channels=64,
                                                          kernel_shape=(
                                                              1, 1, 1),
                                                          channel_factor=channel_factor,
                                                          multi_node_bn_comm=multi_node_bn_comm, )
            # Concat branch 0 through 3

            # Mixed 4f
            # Branch 0
            self.mixed_4f_branch_0_conv3d_0a_1x1 = Unit3D(output_channels=256,
                                                          kernel_shape=(
                                                              1, 1, 1),
                                                          channel_factor=channel_factor,
                                                          multi_node_bn_comm=multi_node_bn_comm, )
            # Branch 1
            self.mixed_4f_branch_1_conv3d_0a_1x1 = Unit3D(output_channels=160,
                                                          kernel_shape=(
                                                              1, 1, 1),
                                                          channel_factor=channel_factor,
                                                          multi_node_bn_comm=multi_node_bn_comm, )
            self.mixed_4f_branch_1_conv3d_0b_3x3 = Unit3D(output_channels=320,
                                                          kernel_shape=(
                                                              3, 3, 3),
                                                          channel_factor=channel_factor,
                                                          multi_node_bn_comm=multi_node_bn_comm, )
            # Branch 2
            self.mixed_4f_branch_2_conv3d_0a_1x1 = Unit3D(output_channels=32,
                                                          kernel_shape=(
                                                              1, 1, 1),
                                                          channel_factor=channel_factor,
                                                          multi_node_bn_comm=multi_node_bn_comm, )
            self.mixed_4f_branch_2_conv3d_0b_3x3 = Unit3D(output_channels=128,
                                                          kernel_shape=(
                                                              3, 3, 3),
                                                          channel_factor=channel_factor,
                                                          multi_node_bn_comm=multi_node_bn_comm, )
            # Branch 3
            # Max pooling 3x3x3 stride 1x1x1
            self.mixed_4f_branch_3_conv3d_0b_1x1 = Unit3D(output_channels=128,
                                                          kernel_shape=(
                                                              1, 1, 1),
                                                          channel_factor=channel_factor,
                                                          multi_node_bn_comm=multi_node_bn_comm, )
            # Concat branch 0 through 3

            # Max pooling 2x2x2 stride 2x2x2

            # Mixed 5b
            # Branch 0
            self.mixed_5b_branch_0_conv3d_0a_1x1 = Unit3D(output_channels=256,
                                                          kernel_shape=(
                                                              1, 1, 1),
                                                          channel_factor=channel_factor,
                                                          multi_node_bn_comm=multi_node_bn_comm, )
            # Branch 1
            self.mixed_5b_branch_1_conv3d_0a_1x1 = Unit3D(output_channels=160,
                                                          kernel_shape=(
                                                              1, 1, 1),
                                                          channel_factor=channel_factor,
                                                          multi_node_bn_comm=multi_node_bn_comm, )
            self.mixed_5b_branch_1_conv3d_0b_3x3 = Unit3D(output_channels=320,
                                                          kernel_shape=(
                                                              3, 3, 3),
                                                          channel_factor=channel_factor,
                                                          multi_node_bn_comm=multi_node_bn_comm, )
            # Branch 2
            self.mixed_5b_branch_2_conv3d_0a_1x1 = Unit3D(output_channels=32,
                                                          kernel_shape=(
                                                              1, 1, 1),
                                                          channel_factor=channel_factor,
                                                          multi_node_bn_comm=multi_node_bn_comm, )
            self.mixed_5b_branch_2_conv3d_0b_3x3 = Unit3D(output_channels=128,
                                                          kernel_shape=(
                                                              3, 3, 3),
                                                          channel_factor=channel_factor,
                                                          multi_node_bn_comm=multi_node_bn_comm, )
            # Branch 3
            # Max pooling 3x3x3 stride 1x1x1
            self.mixed_5b_branch_3_conv3d_0b_1x1 = Unit3D(output_channels=128,
                                                          kernel_shape=(
                                                              1, 1, 1),
                                                          channel_factor=channel_factor,
                                                          multi_node_bn_comm=multi_node_bn_comm, )
            # Concat branch 0 through 3

            # Mixed 5c
            # Branch 0
            self.mixed_5c_branch_0_conv3d_0a_1x1 = Unit3D(output_channels=384,
                                                          kernel_shape=(
                                                              1, 1, 1),
                                                          channel_factor=channel_factor,
                                                          multi_node_bn_comm=multi_node_bn_comm, )
            # Branch 1
            self.mixed_5c_branch_1_conv3d_0a_1x1 = Unit3D(output_channels=192,
                                                          kernel_shape=(
                                                              1, 1, 1),
                                                          channel_factor=channel_factor,
                                                          multi_node_bn_comm=multi_node_bn_comm, )
            self.mixed_5c_branch_1_conv3d_0b_3x3 = Unit3D(output_channels=384,
                                                          kernel_shape=(
                                                              3, 3, 3),
                                                          channel_factor=channel_factor,
                                                          multi_node_bn_comm=multi_node_bn_comm, )
            # Branch 2
            self.mixed_5c_branch_2_conv3d_0a_1x1 = Unit3D(output_channels=48,
                                                          kernel_shape=(
                                                              1, 1, 1),
                                                          channel_factor=channel_factor,
                                                          multi_node_bn_comm=multi_node_bn_comm, )
            self.mixed_5c_branch_2_conv3d_0b_3x3 = Unit3D(output_channels=128,
                                                          kernel_shape=(
                                                              3, 3, 3),
                                                          channel_factor=channel_factor,
                                                          multi_node_bn_comm=multi_node_bn_comm, )
            # Branch 3
            # Max pooling 3x3x3 stride 1x1x1
            self.mixed_5c_branch_3_conv3d_0b_1x1 = Unit3D(output_channels=128,
                                                          kernel_shape=(
                                                              1, 1, 1),
                                                          channel_factor=channel_factor,
                                                          multi_node_bn_comm=multi_node_bn_comm, )
            # Concat branch 0 through 3

            # Logits
            # Average pooling 2x7x7 stride 1x1x1 padding valid (i.e. H and W should become 1)
            # Dropout
            if not self.skip_logits:
                self.logits_conv3d_0c_1x1 = Unit3D(
                    output_channels=self.num_classes,
                    kernel_shape=(1, 1, 1),
                    activation_fn=None,
                    use_batch_norm=False,
                    use_bias=True,
                )

    def max_pool_3d_2a_3x3_block(self, net, layers, outputs):
        net = self.conv3d_1a_7x7(net)
        maybe_add_features(layers, outputs, "conv3d_1a_7x7", net)
        net = max_pooling_3d_same(net, ksize=(self.time_ksizes[1], 3, 3),
                                  strides=(self.time_strides[1], 2, 2))
        maybe_add_features(layers, outputs, "max_pool_3d_2a_3x3", net)

        return net

    def max_pool_3d_3a_3x3_block(self, net, layers, outputs):
        net = self.conv3d_2b_1x1(net)
        maybe_add_features(layers, outputs, "conv3d_2b_1x1", net)
        net = self.conv3d_2c_3x3(net)
        maybe_add_features(layers, outputs, "conv3d_2c_3x3", net)
        net = max_pooling_3d_same(net, ksize=(self.time_ksizes[2], 3, 3),
                                  strides=(self.time_strides[2], 2, 2))
        maybe_add_features(layers, outputs, "max_pool_3d_3a_3x3", net)

        return net

    def max_pool_3d_4a_3x3_block(self, net, layers, outputs):
        # Mixed 3b
        # Branch 0
        mixed_3b_branch_0 = self.mixed_3b_branch_0_conv3d_0a_1x1(net)
        # Branch 1
        mixed_3b_branch_1 = self.mixed_3b_branch_1_conv3d_0a_1x1(net)
        mixed_3b_branch_1 = self.mixed_3b_branch_1_conv3d_0b_3x3(
            mixed_3b_branch_1)
        # Branch 2
        mixed_3b_branch_2 = self.mixed_3b_branch_2_conv3d_0a_1x1(net)
        mixed_3b_branch_2 = self.mixed_3b_branch_2_conv3d_0b_3x3(
            mixed_3b_branch_2)
        # Branch 3
        mixed_3b_branch_3 = max_pooling_3d_same(net, ksize=(3, 3, 3),
                                                strides=(1, 1, 1))
        mixed_3b_branch_3 = self.mixed_3b_branch_3_conv3d_0b_1x1(
            mixed_3b_branch_3)
        net = F.concat([
            mixed_3b_branch_0,
            mixed_3b_branch_1,
            mixed_3b_branch_2,
            mixed_3b_branch_3
        ], axis=1)
        maybe_add_features(layers, outputs, "mixed_3b", net)
        # Mixed 3c
        # Branch 0
        mixed_3c_branch_0 = self.mixed_3c_branch_0_conv3d_0a_1x1(net)
        # Branch 1
        mixed_3c_branch_1 = self.mixed_3c_branch_1_conv3d_0a_1x1(net)
        mixed_3c_branch_1 = self.mixed_3c_branch_1_conv3d_0b_3x3(
            mixed_3c_branch_1)
        # Branch 2
        mixed_3c_branch_2 = self.mixed_3c_branch_2_conv3d_0a_1x1(net)
        mixed_3c_branch_2 = self.mixed_3c_branch_2_conv3d_0b_3x3(
            mixed_3c_branch_2)
        # Branch 3
        mixed_3c_branch_3 = max_pooling_3d_same(net, ksize=(3, 3, 3),
                                                strides=(1, 1, 1))
        mixed_3c_branch_3 = self.mixed_3c_branch_3_conv3d_0b_1x1(
            mixed_3c_branch_3)
        net = F.concat([
            mixed_3c_branch_0,
            mixed_3c_branch_1,
            mixed_3c_branch_2,
            mixed_3c_branch_3
        ], axis=1)
        maybe_add_features(layers, outputs, "mixed_3c", net)
        net = max_pooling_3d_same(net, ksize=(self.time_ksizes[3], 3, 3),
                                  strides=(self.time_strides[3], 2, 2))
        maybe_add_features(layers, outputs, "max_pool_3d_4a_3x3", net)

        return net

    def max_pool_3d_5a_2x2_block(self, net, layers, outputs):
        # Mixed 4b
        # Branch 0
        mixed_4b_branch_0 = self.mixed_4b_branch_0_conv3d_0a_1x1(net)
        # Branch 1
        mixed_4b_branch_1 = self.mixed_4b_branch_1_conv3d_0a_1x1(net)
        mixed_4b_branch_1 = self.mixed_4b_branch_1_conv3d_0b_3x3(
            mixed_4b_branch_1)
        # Branch 2
        mixed_4b_branch_2 = self.mixed_4b_branch_2_conv3d_0a_1x1(net)
        mixed_4b_branch_2 = self.mixed_4b_branch_2_conv3d_0b_3x3(
            mixed_4b_branch_2)
        # Branch 3
        mixed_4b_branch_3 = max_pooling_3d_same(net, ksize=(3, 3, 3),
                                                strides=(1, 1, 1))
        mixed_4b_branch_3 = self.mixed_4b_branch_3_conv3d_0b_1x1(
            mixed_4b_branch_3)
        net = F.concat([
            mixed_4b_branch_0,
            mixed_4b_branch_1,
            mixed_4b_branch_2,
            mixed_4b_branch_3
        ], axis=1)
        maybe_add_features(layers, outputs, "mixed_4b", net)

        # Mixed 4c
        # Branch 0
        mixed_4c_branch_0 = self.mixed_4c_branch_0_conv3d_0a_1x1(net)
        # Branch 1
        mixed_4c_branch_1 = self.mixed_4c_branch_1_conv3d_0a_1x1(net)
        mixed_4c_branch_1 = self.mixed_4c_branch_1_conv3d_0b_3x3(
            mixed_4c_branch_1)
        # Branch 2
        mixed_4c_branch_2 = self.mixed_4c_branch_2_conv3d_0a_1x1(net)
        mixed_4c_branch_2 = self.mixed_4c_branch_2_conv3d_0b_3x3(
            mixed_4c_branch_2)
        # Branch 3
        mixed_4c_branch_3 = max_pooling_3d_same(net, ksize=(3, 3, 3),
                                                strides=(1, 1, 1))
        mixed_4c_branch_3 = self.mixed_4c_branch_3_conv3d_0b_1x1(
            mixed_4c_branch_3)
        net = F.concat([
            mixed_4c_branch_0,
            mixed_4c_branch_1,
            mixed_4c_branch_2,
            mixed_4c_branch_3
        ], axis=1)
        maybe_add_features(layers, outputs, "mixed_4c", net)

        # Mixed 4d
        # Branch 0
        mixed_4d_branch_0 = self.mixed_4d_branch_0_conv3d_0a_1x1(net)
        # Branch 1
        mixed_4d_branch_1 = self.mixed_4d_branch_1_conv3d_0a_1x1(net)
        mixed_4d_branch_1 = self.mixed_4d_branch_1_conv3d_0b_3x3(
            mixed_4d_branch_1)
        # Branch 2
        mixed_4d_branch_2 = self.mixed_4d_branch_2_conv3d_0a_1x1(net)
        mixed_4d_branch_2 = self.mixed_4d_branch_2_conv3d_0b_3x3(
            mixed_4d_branch_2)
        # Branch 3
        mixed_4d_branch_3 = max_pooling_3d_same(net, ksize=(3, 3, 3),
                                                strides=(1, 1, 1))
        mixed_4d_branch_3 = self.mixed_4d_branch_3_conv3d_0b_1x1(
            mixed_4d_branch_3)
        net = F.concat([
            mixed_4d_branch_0,
            mixed_4d_branch_1,
            mixed_4d_branch_2,
            mixed_4d_branch_3
        ], axis=1)
        maybe_add_features(layers, outputs, "mixed_4d", net)

        # Mixed 4e
        # Branch 0
        mixed_4e_branch_0 = self.mixed_4e_branch_0_conv3d_0a_1x1(net)
        # Branch 1
        mixed_4e_branch_1 = self.mixed_4e_branch_1_conv3d_0a_1x1(net)
        mixed_4e_branch_1 = self.mixed_4e_branch_1_conv3d_0b_3x3(
            mixed_4e_branch_1)
        # Branch 2
        mixed_4e_branch_2 = self.mixed_4e_branch_2_conv3d_0a_1x1(net)
        mixed_4e_branch_2 = self.mixed_4e_branch_2_conv3d_0b_3x3(
            mixed_4e_branch_2)
        # Branch 3
        mixed_4e_branch_3 = max_pooling_3d_same(net, ksize=(3, 3, 3),
                                                strides=(1, 1, 1))
        mixed_4e_branch_3 = self.mixed_4e_branch_3_conv3d_0b_1x1(
            mixed_4e_branch_3)
        net = F.concat([
            mixed_4e_branch_0,
            mixed_4e_branch_1,
            mixed_4e_branch_2,
            mixed_4e_branch_3
        ], axis=1)
        maybe_add_features(layers, outputs, "mixed_4e", net)

        # Mixed 4f
        # Branch 0
        mixed_4f_branch_0 = self.mixed_4f_branch_0_conv3d_0a_1x1(net)
        # Branch 1
        mixed_4f_branch_1 = self.mixed_4f_branch_1_conv3d_0a_1x1(net)
        mixed_4f_branch_1 = self.mixed_4f_branch_1_conv3d_0b_3x3(
            mixed_4f_branch_1)
        # Branch 2
        mixed_4f_branch_2 = self.mixed_4f_branch_2_conv3d_0a_1x1(net)
        mixed_4f_branch_2 = self.mixed_4f_branch_2_conv3d_0b_3x3(
            mixed_4f_branch_2)
        # Branch 3
        mixed_4f_branch_3 = max_pooling_3d_same(net, ksize=(3, 3, 3),
                                                strides=(1, 1, 1))
        mixed_4f_branch_3 = self.mixed_4f_branch_3_conv3d_0b_1x1(
            mixed_4f_branch_3)
        net = F.concat([
            mixed_4f_branch_0,
            mixed_4f_branch_1,
            mixed_4f_branch_2,
            mixed_4f_branch_3
        ], axis=1)
        maybe_add_features(layers, outputs, "mixed_4f", net)
        net = max_pooling_3d_same(net, ksize=(self.time_ksizes[4], 2, 2),
                                  strides=(self.time_strides[4], 2, 2))
        maybe_add_features(layers, outputs, "max_pool_3d_5a_2x2", net)

        return net

    def avg_pool_block(self, net, layers, outputs):
        # Mixed 5b
        # Branch 0
        mixed_5b_branch_0 = self.mixed_5b_branch_0_conv3d_0a_1x1(net)
        # Branch 1
        mixed_5b_branch_1 = self.mixed_5b_branch_1_conv3d_0a_1x1(net)
        mixed_5b_branch_1 = self.mixed_5b_branch_1_conv3d_0b_3x3(
            mixed_5b_branch_1)
        # Branch 2
        mixed_5b_branch_2 = self.mixed_5b_branch_2_conv3d_0a_1x1(net)
        mixed_5b_branch_2 = self.mixed_5b_branch_2_conv3d_0b_3x3(
            mixed_5b_branch_2)
        # Branch 3
        mixed_5b_branch_3 = max_pooling_3d_same(net, ksize=(3, 3, 3),
                                                strides=(1, 1, 1))
        mixed_5b_branch_3 = self.mixed_5b_branch_3_conv3d_0b_1x1(
            mixed_5b_branch_3)
        net = F.concat([
            mixed_5b_branch_0,
            mixed_5b_branch_1,
            mixed_5b_branch_2,
            mixed_5b_branch_3
        ], axis=1)
        maybe_add_features(layers, outputs, "mixed_5b", net)
        # Mixed 5c
        # Branch 0
        mixed_5c_branch_0 = self.mixed_5c_branch_0_conv3d_0a_1x1(net)
        # Branch 1
        mixed_5c_branch_1 = self.mixed_5c_branch_1_conv3d_0a_1x1(net)
        mixed_5c_branch_1 = self.mixed_5c_branch_1_conv3d_0b_3x3(
            mixed_5c_branch_1)
        # Branch 2
        mixed_5c_branch_2 = self.mixed_5c_branch_2_conv3d_0a_1x1(net)
        mixed_5c_branch_2 = self.mixed_5c_branch_2_conv3d_0b_3x3(
            mixed_5c_branch_2)
        # Branch 3
        mixed_5c_branch_3 = max_pooling_3d_same(net, ksize=(3, 3, 3),
                                                strides=(1, 1, 1))
        mixed_5c_branch_3 = self.mixed_5c_branch_3_conv3d_0b_1x1(
            mixed_5c_branch_3)
        net = F.concat([
            mixed_5c_branch_0,
            mixed_5c_branch_1,
            mixed_5c_branch_2,
            mixed_5c_branch_3
        ], axis=1)
        maybe_add_features(layers, outputs, "mixed_5c", net)

        # Logits
        f_height, f_width = net.shape[-2:]
        net = F.average_pooling_3d(net, ksize=(2, f_height, f_width),
                                   stride=(1, 1, 1))
        maybe_add_features(layers, outputs, "avg_pool", net)

        return net

    def __call__(self, inputs, layers=None):
        if layers is None:
            layers = ["averaged_logits"]
        outputs = {}

        net = self.max_pool_3d_2a_3x3_block(inputs, layers, outputs)

        net = self.max_pool_3d_3a_3x3_block(net, layers, outputs)

        net = self.max_pool_3d_4a_3x3_block(net, layers, outputs)

        net = self.max_pool_3d_5a_2x2_block(net, layers, outputs)

        net = self.avg_pool_block(net, layers, outputs)

        if not self.skip_logits:
            net = F.dropout(net, 1. - self.dropout_keep_prob)
            net = self.logits_conv3d_0c_1x1(net)
            maybe_add_features(layers, outputs, "logits", net)
            net = F.squeeze(F.mean(net, axis=2), axis=(2, 3))
            maybe_add_features(layers, outputs, "averaged_logits", net)

        if len(outputs) == 1:
            return list(outputs.values())[0]
        else:
            return outputs

