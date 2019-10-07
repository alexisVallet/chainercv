"""
Implementation of the I3D model, using pre-trained weights from deepmind.
"""
from functools import partial

import chainer
import chainer.links as L
from chainer import functions as F
from chainer.functions import pad
from chainer.links import Convolution3D
from chainer.utils import conv
import chainermn
import numpy as np

from chainercv.links import PickableSequentialChain


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
                 use_bias=False, multi_node_bn_comm=None):
        super(Unit3D, self).__init__()
        self.activation_fn = activation_fn
        self.use_batch_norm = use_batch_norm
        self.use_bias = use_bias

        with self.init_scope():
            self.conv3d = TFConvolution3D(
                in_channels=None,
                out_channels=output_channels,
                ksize=kernel_shape,
                stride=stride,
                nobias=not use_bias,
            )
            if use_batch_norm:
                if multi_node_bn_comm is None:
                    self.bn = L.BatchNormalization(
                        size=output_channels,
                        decay=0.999,
                        eps=1e-3,
                        use_gamma=False,
                    )
                else:
                    self.bn = chainermn.links.MultiNodeBatchNormalization(
                        size=output_channels,
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


def i3d_global_average_pooling(inputs):
    f_height, f_width = inputs.shape[-2:]

    return F.average_pooling_3d(inputs, ksize=(2, f_height, f_width),
                                stride=(1, 1, 1))

def maybe_add_features(layers, outputs, layer_name, x):
    if layer_name in layers:
        outputs[layer_name] = x


class I3DInceptionBlock(chainer.Chain):
    def __init__(self, branch_0_channels, branch_1_channels, branch_2_channels,
                 branch_3_channels, multi_node_bn_comm=None):
        super(I3DInceptionBlock, self).__init__()
        with self.init_scope():
            self.branch_0_conv3d_0a_1x1 = Unit3D(
                output_channels=branch_0_channels, kernel_shape=(1, 1, 1),
                multi_node_bn_comm=multi_node_bn_comm)
            # Branch 1
            self.branch_1_conv3d_0a_1x1 = Unit3D(
                output_channels=branch_1_channels[0], kernel_shape=(1, 1, 1),
                multi_node_bn_comm=multi_node_bn_comm)
            self.branch_1_conv3d_0b_3x3 = Unit3D(
                output_channels=branch_1_channels[1], kernel_shape=(3, 3, 3),
                multi_node_bn_comm=multi_node_bn_comm)
            # Branch 2
            self.branch_2_conv3d_0a_1x1 = Unit3D(
                output_channels=branch_2_channels[0], kernel_shape=(1, 1, 1),
                multi_node_bn_comm=multi_node_bn_comm)
            self.branch_2_conv3d_0b_3x3 = Unit3D(
                output_channels=branch_2_channels[1], kernel_shape=(3, 3, 3),
                multi_node_bn_comm=multi_node_bn_comm)
            # Branch 3
            # Max pooling 3x3x3 stride 1x1x1
            self.branch_3_conv3d_0b_1x1 = Unit3D(
                output_channels=branch_3_channels, kernel_shape=(1, 1, 1),
                multi_node_bn_comm=multi_node_bn_comm)

    def __call__(self, net):
        branch_0 = self.branch_0_conv3d_0a_1x1(net)
        # Branch 1
        branch_1 = self.branch_1_conv3d_0a_1x1(net)
        branch_1 = self.branch_1_conv3d_0b_3x3(branch_1)
        # Branch 2
        branch_2 = self.branch_2_conv3d_0a_1x1(net)
        branch_2 = self.branch_2_conv3d_0b_3x3(branch_2)
        # Branch 3
        branch_3 = max_pooling_3d_same(net, ksize=(3, 3, 3), strides=(1, 1, 1))
        branch_3 = self.branch_3_conv3d_0b_1x1(branch_3)
        net = F.concat([branch_0, branch_1, branch_2, branch_3], axis=1)

        return net


class I3D(PickableSequentialChain):
    def __init__(self, num_classes, dropout_keep_prob,
                 time_strides=(2, 1, 1, 2, 2,), time_ksizes=(7, 1, 1, 3, 2,),
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
            self.conv3d_1a_7x7 = Unit3D(output_channels=64,
                                        kernel_shape=(time_ksizes[0], 7, 7),
                                        stride=(time_strides[0], 2, 2),
                                        multi_node_bn_comm=multi_node_bn_comm)
            # Max pooling 1x3x3
            self.max_pool_3d_2a_3x3 = partial(
                max_pooling_3d_same, ksize=(self.time_ksizes[1], 3, 3),
                strides=(self.time_strides[1], 2, 2))
            self.conv3d_2b_1x1 = Unit3D(output_channels=64,
                                        kernel_shape=(1, 1, 1),
                                        multi_node_bn_comm=multi_node_bn_comm)
            self.conv3d_2c_3x3 = Unit3D(output_channels=192,
                                        kernel_shape=(3, 3, 3),
                                        multi_node_bn_comm=multi_node_bn_comm)
            # Max pooling 1x3x3
            self.max_pool_3d_3a_3x3 = partial(
                max_pooling_3d_same, ksize=(self.time_ksizes[2], 3, 3),
                strides=(self.time_strides[2], 2, 2))
            # Mixed 3b
            self.mixed_3b = I3DInceptionBlock(branch_0_channels=64,
                                              branch_1_channels=(96, 128),
                                              branch_2_channels=(16, 32),
                                              branch_3_channels=32,
                                              multi_node_bn_comm=multi_node_bn_comm)

            # Mixed 3c
            self.mixed_3c = I3DInceptionBlock(branch_0_channels=128,
                                              branch_1_channels=(128, 192),
                                              branch_2_channels=(32, 96),
                                              branch_3_channels=64,
                                              multi_node_bn_comm=multi_node_bn_comm)

            # Max pooling 3x3x3 stride 2x2x2
            self.max_pool_3d_4a_3x3 = partial(
                max_pooling_3d_same, ksize=(self.time_ksizes[3], 3, 3),
                strides=(self.time_strides[3], 2, 2))
            # Mixed 4b
            self.mixed_4b = I3DInceptionBlock(branch_0_channels=192,
                                              branch_1_channels=(96, 208),
                                              branch_2_channels=(16, 48),
                                              branch_3_channels=64,
                                              multi_node_bn_comm=multi_node_bn_comm)

            # Mixed 4c
            self.mixed_4c = I3DInceptionBlock(branch_0_channels=160,
                                              branch_1_channels=(112, 224),
                                              branch_2_channels=(24, 64),
                                              branch_3_channels=64,
                                              multi_node_bn_comm=multi_node_bn_comm)

            # Mixed 4d
            self.mixed_4d = I3DInceptionBlock(branch_0_channels=128,
                                              branch_1_channels=(128, 256),
                                              branch_2_channels=(24, 64),
                                              branch_3_channels=64,
                                              multi_node_bn_comm=multi_node_bn_comm)

            # Mixed 4e
            self.mixed_4e = I3DInceptionBlock(branch_0_channels=112,
                                              branch_1_channels=(144, 288),
                                              branch_2_channels=(32, 64),
                                              branch_3_channels=64,
                                              multi_node_bn_comm=multi_node_bn_comm)

            # Mixed 4f
            self.mixed_4f = I3DInceptionBlock(branch_0_channels=256,
                                              branch_1_channels=(160, 320),
                                              branch_2_channels=(32, 128),
                                              branch_3_channels=128,
                                              multi_node_bn_comm=multi_node_bn_comm)

            # Max pooling 2x2x2 stride 2x2x2
            self.max_pool_3d_5a_2x2 = partial(
                max_pooling_3d_same, ksize=(self.time_ksizes[4], 2, 2),
                strides=(self.time_strides[4], 2, 2))
            # Mixed 5b
            self.mixed_5b = I3DInceptionBlock(branch_0_channels=256,
                                              branch_1_channels=(160, 320),
                                              branch_2_channels=(32, 128),
                                              branch_3_channels=128,
                                              multi_node_bn_comm=multi_node_bn_comm)

            # Mixed 5c
            self.mixed_5c = I3DInceptionBlock(branch_0_channels=384,
                                              branch_1_channels=(192, 384),
                                              branch_2_channels=(48, 128),
                                              branch_3_channels=128,
                                              multi_node_bn_comm=multi_node_bn_comm)

            # Logits
            # Average pooling 2x7x7 stride 1x1x1 padding valid
            # (i.e. H and W should become 1)
            self.avg_pool = i3d_global_average_pooling

            if not self.skip_logits:
                # Dropout
                self.dropout = partial(
                    F.dropout, ratio=1 - self.dropout_keep_prob)
                self.logits_conv3d_0c_1x1 = Unit3D(
                    output_channels=self.num_classes, kernel_shape=(1, 1, 1),
                    activation_fn=None, use_batch_norm=False, use_bias=True)
                # Average pooling, removing unnecessary spatial dimensions.
                self.time_mean_pool = partial(F.mean, axis=2)
                self.averaged_logits = partial(F.squeeze, axis=(2, 3))
