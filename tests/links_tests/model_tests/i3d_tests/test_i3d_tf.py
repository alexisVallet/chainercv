"""
Tests for I3D.
"""
import chainer
chainer.config.use_cudnn = 'always'
import numpy as np
import cupy as cp
import os
import cv2
import re
from hypothesis import given, settings
from hypothesis.strategies import floats, integers, tuples, data
from hypothesis.extra.numpy import arrays
from pytest import fixture
import sonnet as snt
import tensorflow as tf
import tempfile

import chainercv.links.model.i3d.i3d as i3d
from chainercv.datasets.kinetics.kinetics_dataset import KineticsDataset
from examples.i3d.tfckpt2npz import load_tensorflow_checkpoint, \
    unit3d_load_tensorflow_weights, tf_checkpoint_to_npz

_TEST_VIDEO_DIRECTORY = os.path.join(
    'data',
    'kinetics_val'
)
_TEST_LABEL_MAP = os.path.join(
    'models',
    'rgb_scratch_kin600',
    'label_map_600.txt'
)
_TEST_CHECKPOINT = os.path.join(
    'models',
    'rgb_scratch_kin600',
    'model.ckpt'
)


def test_simple():
    with chainer.using_config('train', True):
        chainer.cuda.get_device_from_id(0).use()
        model = i3d.I3D(600, 0.5)
        load_tensorflow_checkpoint(model, _TEST_CHECKPOINT)
        model.to_gpu()
        dummy_batch = np.random.uniform(-1, 1, (8, 3, 64, 64, 64)).astype(np.float32)
        dummy_batch = chainer.Variable(chainer.backends.cuda.to_gpu(dummy_batch, device=0))
        features = model(dummy_batch, layers=(
            "conv3d_1a_7x7",
            "max_pool_3d_2a_3x3",
            "max_pool_3d_3a_3x3",
            "mixed_3b",
            "max_pool_3d_4a_3x3",
            "mixed_3b",
            "mixed_4b",
            "mixed_4c",
            "mixed_4d",
            "mixed_4e",
            "mixed_4f",
            "max_pool_3d_5a_2x2",
            "avg_pool",
            "logits",
        ))
        for k, v in features.items():
            v = cp.asnumpy(v.data)
            print(k, v.shape, v.mean(), v.std(), v.min(), v.max())


@ fixture()
def tf_conv3d_fixture():
    _x = tf.placeholder(dtype=tf.float32, shape=(None, None, None, None, None))
    _w = tf.placeholder(dtype=tf.float32, shape=(None, None, None, None, None))
    _b = tf.placeholder(dtype=tf.float32, shape=(1, 1, 1, 1,  None))
    _beta = tf.placeholder(dtype=tf.float32, shape=(1, 1, 1, 1, None))
    _moving_mean = tf.placeholder(dtype=tf.float32, shape=(1, 1, 1, 1, None))
    _moving_variance = tf.placeholder(dtype=tf.float32, shape=(1, 1, 1, 1, None))
    _y = tf.nn.conv3d(_x, _w, (1, 1, 1, 1, 1), padding='SAME') + _b
    _y = tf.nn.batch_normalization(
        _y,
        mean=_moving_mean,
        variance=_moving_variance,
        offset=_beta,
        scale=None,
        variance_epsilon=1e-3,
    )
    sess = tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(
        per_process_gpu_memory_fraction=0.3
    )))

    def _conv3d(x, w, b, beta, moving_mean, moving_variance):
        return sess.run(_y, feed_dict={
            _x: x,
            _w: w,
            _b: b,
            _beta: beta,
            _moving_mean: moving_mean,
            _moving_variance: moving_variance,
        })
    return _conv3d


@settings(deadline=None)
@given(data=data())
def test_load_tensorflow_weights(data, tf_conv3d_fixture):
    stride = 1
    kdim = data.draw(integers(min_value=1, max_value=7))
    in_channels = data.draw(integers(min_value=1, max_value=16))
    out_channels = data.draw(integers(min_value=1, max_value=16))
    crop_size = data.draw(integers(min_value=1, max_value=16))
    seq_length = data.draw(integers(min_value=1, max_value=8))
    batch_size = data.draw(integers(min_value=1, max_value=4))
    scalar_strat = floats(min_value=-1, max_value=1, allow_nan=False, allow_infinity=False)
    x_tf = data.draw(arrays(shape=(batch_size, seq_length, crop_size, crop_size, in_channels),
                            elements=scalar_strat, dtype=np.float32))
    w_tf = data.draw(arrays(shape=(kdim, kdim, kdim, in_channels, out_channels),
                            elements=scalar_strat, dtype=np.float32))
    b_tf = data.draw(arrays(shape=(1, 1, 1, 1, out_channels), elements=scalar_strat, dtype=np.float32))
    beta_tf = data.draw(arrays(shape=(1, 1, 1, 1, out_channels), elements=scalar_strat, dtype=np.float32))
    moving_mean_tf = data.draw(arrays(shape=(1, 1, 1, 1, out_channels), elements=scalar_strat, dtype=np.float32))
    moving_variance_tf = data.draw(arrays(shape=(1, 1, 1, 1, out_channels),
                                          elements=floats(min_value=0.001, max_value=1.999, allow_nan=False, allow_infinity=False), dtype=np.float32))
    y_tf = tf_conv3d_fixture(x_tf, w_tf, b_tf, beta_tf, moving_mean_tf, moving_variance_tf)

    with chainer.using_config('train', False):
        unit3d = i3d.Unit3D(
            output_channels=out_channels,
            kernel_shape=(kdim, kdim, kdim),
            stride=(stride, stride, stride),
            use_batch_norm=True,
            use_bias=True,
            activation_fn=None,
        )
        unit3d_load_tensorflow_weights(unit3d, weights=w_tf, bias=b_tf, beta=beta_tf, moving_mean=moving_mean_tf,
                                       moving_variance=moving_variance_tf)
        unit3d.to_gpu(0)
        x_ch = chainer.Variable(np.moveaxis(x_tf, 4, 1))
        x_ch.to_gpu(0)
        y_ch = unit3d(x_ch)
    actual_w = np.moveaxis(
        cp.asnumpy(unit3d.conv3d.conv.W.data),
        1,
        -1
    )
    actual_w = np.moveaxis(
        actual_w,
        0,
        -1
    )
    actual_b = np.reshape(
        cp.asnumpy(unit3d.conv3d.conv.b.data),
        (1, 1, 1, 1, out_channels)
    )
    np.testing.assert_equal(actual_w, w_tf)
    np.testing.assert_equal(actual_b, b_tf)

    # Checking that the output is near identical.
    y_ch_tf = np.moveaxis(cp.asnumpy(y_ch.data), 1, 4)
    np.testing.assert_allclose(y_ch_tf, y_tf, rtol=0.001, atol=0.001)


# Include the original Tensorflow I3D model for testing purposes only.
class Unit3D(snt.AbstractModule):
  """Basic unit containing Conv3D + BatchNorm + non-linearity."""

  def __init__(self, output_channels,
               kernel_shape=(1, 1, 1),
               stride=(1, 1, 1),
               activation_fn=tf.nn.relu,
               use_batch_norm=True,
               use_bias=False,
               name='unit_3d'):
    """Initializes Unit3D module."""
    super(Unit3D, self).__init__(name=name)
    self._output_channels = output_channels
    self._kernel_shape = kernel_shape
    self._stride = stride
    self._use_batch_norm = use_batch_norm
    self._activation_fn = activation_fn
    self._use_bias = use_bias

  def _build(self, inputs, is_training):
    """Connects the module to inputs.
    Args:
      inputs: Inputs to the Unit3D component.
      is_training: whether to use training mode for snt.BatchNorm (boolean).
    Returns:
      Outputs from the module.
    """
    net = snt.Conv3D(output_channels=self._output_channels,
                     kernel_shape=self._kernel_shape,
                     stride=self._stride,
                     padding=snt.SAME,
                     use_bias=self._use_bias)(inputs)
    if self._use_batch_norm:
      bn = snt.BatchNorm()
      net = bn(net, is_training=is_training, test_local_stats=False)
    if self._activation_fn is not None:
      net = self._activation_fn(net)
    return net


class I3D(snt.AbstractModule):
  """Inception-v1 I3D architecture.
  The model is introduced in:
    Quo Vadis, Action Recognition? A New Model and the Kinetics Dataset
    Joao Carreira, Andrew Zisserman
    https://arxiv.org/pdf/1705.07750v1.pdf.
  See also the Inception architecture, introduced in:
    Going deeper with convolutions
    Christian Szegedy, Wei Liu, Yangqing Jia, Pierre Sermanet, Scott Reed,
    Dragomir Anguelov, Dumitru Erhan, Vincent Vanhoucke, Andrew Rabinovich.
    http://arxiv.org/pdf/1409.4842v1.pdf.
  """

  # Endpoints of the model in order. During construction, all the endpoints up
  # to a designated `final_endpoint` are returned in a dictionary as the
  # second return value.
  VALID_ENDPOINTS = (
      'Conv3d_1a_7x7',
      'MaxPool3d_2a_3x3',
      'Conv3d_2b_1x1',
      'Conv3d_2c_3x3',
      'MaxPool3d_3a_3x3',
      'Mixed_3b',
      'Mixed_3c',
      'MaxPool3d_4a_3x3',
      'Mixed_4b',
      'Mixed_4c',
      'Mixed_4d',
      'Mixed_4e',
      'Mixed_4f',
      'MaxPool3d_5a_2x2',
      'Mixed_5b',
      'Mixed_5c',
      'Logits',
      'Predictions',
  )

  def __init__(self, num_classes=400, spatial_squeeze=True,
               final_endpoint='Logits', name='inception_i3d'):
    """Initializes I3D model instance.
    Args:
      num_classes: The number of outputs in the logit layer (default 400, which
          matches the Kinetics dataset).
      spatial_squeeze: Whether to squeeze the spatial dimensions for the logits
          before returning (default True).
      final_endpoint: The model contains many possible endpoints.
          `final_endpoint` specifies the last endpoint for the model to be built
          up to. In addition to the output at `final_endpoint`, all the outputs
          at endpoints up to `final_endpoint` will also be returned, in a
          dictionary. `final_endpoint` must be one of
          I3D.VALID_ENDPOINTS (default 'Logits').
      name: A string (optional). The name of this module.
    Raises:
      ValueError: if `final_endpoint` is not recognized.
    """

    if final_endpoint not in self.VALID_ENDPOINTS:
      raise ValueError('Unknown final endpoint %s' % final_endpoint)

    super(I3D, self).__init__(name=name)
    self._num_classes = num_classes
    self._spatial_squeeze = spatial_squeeze
    self._final_endpoint = final_endpoint

  def _build(self, inputs, is_training, dropout_keep_prob=1.0):
    """Connects the model to inputs.
    Args:
      inputs: Inputs to the model, which should have dimensions
          `batch_size` x `num_frames` x 224 x 224 x `num_channels`.
      is_training: whether to use training mode for snt.BatchNorm (boolean).
      dropout_keep_prob: Probability for the tf.nn.dropout layer (float in
          [0, 1)).
    Returns:
      A tuple consisting of:
        1. Network output at location `self._final_endpoint`.
        2. Dictionary containing all endpoints up to `self._final_endpoint`,
           indexed by endpoint name.
    Raises:
      ValueError: if `self._final_endpoint` is not recognized.
    """
    if self._final_endpoint not in self.VALID_ENDPOINTS:
      raise ValueError('Unknown final endpoint %s' % self._final_endpoint)

    net = inputs
    end_points = {}
    end_point = 'Conv3d_1a_7x7'
    net = Unit3D(output_channels=64, kernel_shape=[7, 7, 7],
                 stride=[2, 2, 2], name=end_point)(net, is_training=is_training)
    end_points[end_point] = net
    if self._final_endpoint == end_point: return net, end_points
    end_point = 'MaxPool3d_2a_3x3'
    net = tf.nn.max_pool3d(net, ksize=[1, 1, 3, 3, 1], strides=[1, 1, 2, 2, 1],
                           padding=snt.SAME, name=end_point)
    end_points[end_point] = net
    if self._final_endpoint == end_point: return net, end_points
    end_point = 'Conv3d_2b_1x1'
    net = Unit3D(output_channels=64, kernel_shape=[1, 1, 1],
                 name=end_point)(net, is_training=is_training)
    end_points[end_point] = net
    if self._final_endpoint == end_point: return net, end_points
    end_point = 'Conv3d_2c_3x3'
    net = Unit3D(output_channels=192, kernel_shape=[3, 3, 3],
                 name=end_point)(net, is_training=is_training)
    end_points[end_point] = net
    if self._final_endpoint == end_point: return net, end_points
    end_point = 'MaxPool3d_3a_3x3'
    net = tf.nn.max_pool3d(net, ksize=[1, 1, 3, 3, 1], strides=[1, 1, 2, 2, 1],
                           padding=snt.SAME, name=end_point)
    end_points[end_point] = net
    if self._final_endpoint == end_point: return net, end_points

    end_point = 'Mixed_3b'
    with tf.variable_scope(end_point):
      with tf.variable_scope('Branch_0'):
        branch_0 = Unit3D(output_channels=64, kernel_shape=[1, 1, 1],
                          name='Conv3d_0a_1x1')(net, is_training=is_training)
      with tf.variable_scope('Branch_1'):
        branch_1 = Unit3D(output_channels=96, kernel_shape=[1, 1, 1],
                          name='Conv3d_0a_1x1')(net, is_training=is_training)
        branch_1 = Unit3D(output_channels=128, kernel_shape=[3, 3, 3],
                          name='Conv3d_0b_3x3')(branch_1,
                                                is_training=is_training)
      with tf.variable_scope('Branch_2'):
        branch_2 = Unit3D(output_channels=16, kernel_shape=[1, 1, 1],
                          name='Conv3d_0a_1x1')(net, is_training=is_training)
        branch_2 = Unit3D(output_channels=32, kernel_shape=[3, 3, 3],
                          name='Conv3d_0b_3x3')(branch_2,
                                                is_training=is_training)
      with tf.variable_scope('Branch_3'):
        branch_3 = tf.nn.max_pool3d(net, ksize=[1, 3, 3, 3, 1],
                                    strides=[1, 1, 1, 1, 1], padding=snt.SAME,
                                    name='MaxPool3d_0a_3x3')
        branch_3 = Unit3D(output_channels=32, kernel_shape=[1, 1, 1],
                          name='Conv3d_0b_1x1')(branch_3,
                                                is_training=is_training)

      net = tf.concat([branch_0, branch_1, branch_2, branch_3], 4)
    end_points[end_point] = net
    if self._final_endpoint == end_point: return net, end_points

    end_point = 'Mixed_3c'
    with tf.variable_scope(end_point):
      with tf.variable_scope('Branch_0'):
        branch_0 = Unit3D(output_channels=128, kernel_shape=[1, 1, 1],
                          name='Conv3d_0a_1x1')(net, is_training=is_training)
      with tf.variable_scope('Branch_1'):
        branch_1 = Unit3D(output_channels=128, kernel_shape=[1, 1, 1],
                          name='Conv3d_0a_1x1')(net, is_training=is_training)
        branch_1 = Unit3D(output_channels=192, kernel_shape=[3, 3, 3],
                          name='Conv3d_0b_3x3')(branch_1,
                                                is_training=is_training)
      with tf.variable_scope('Branch_2'):
        branch_2 = Unit3D(output_channels=32, kernel_shape=[1, 1, 1],
                          name='Conv3d_0a_1x1')(net, is_training=is_training)
        branch_2 = Unit3D(output_channels=96, kernel_shape=[3, 3, 3],
                          name='Conv3d_0b_3x3')(branch_2,
                                                is_training=is_training)
      with tf.variable_scope('Branch_3'):
        branch_3 = tf.nn.max_pool3d(net, ksize=[1, 3, 3, 3, 1],
                                    strides=[1, 1, 1, 1, 1], padding=snt.SAME,
                                    name='MaxPool3d_0a_3x3')
        branch_3 = Unit3D(output_channels=64, kernel_shape=[1, 1, 1],
                          name='Conv3d_0b_1x1')(branch_3,
                                                is_training=is_training)
      net = tf.concat([branch_0, branch_1, branch_2, branch_3], 4)
    end_points[end_point] = net
    if self._final_endpoint == end_point: return net, end_points

    end_point = 'MaxPool3d_4a_3x3'
    net = tf.nn.max_pool3d(net, ksize=[1, 3, 3, 3, 1], strides=[1, 2, 2, 2, 1],
                           padding=snt.SAME, name=end_point)
    end_points[end_point] = net
    if self._final_endpoint == end_point: return net, end_points

    end_point = 'Mixed_4b'
    with tf.variable_scope(end_point):
      with tf.variable_scope('Branch_0'):
        branch_0 = Unit3D(output_channels=192, kernel_shape=[1, 1, 1],
                          name='Conv3d_0a_1x1')(net, is_training=is_training)
      with tf.variable_scope('Branch_1'):
        branch_1 = Unit3D(output_channels=96, kernel_shape=[1, 1, 1],
                          name='Conv3d_0a_1x1')(net, is_training=is_training)
        branch_1 = Unit3D(output_channels=208, kernel_shape=[3, 3, 3],
                          name='Conv3d_0b_3x3')(branch_1,
                                                is_training=is_training)
      with tf.variable_scope('Branch_2'):
        branch_2 = Unit3D(output_channels=16, kernel_shape=[1, 1, 1],
                          name='Conv3d_0a_1x1')(net, is_training=is_training)
        branch_2 = Unit3D(output_channels=48, kernel_shape=[3, 3, 3],
                          name='Conv3d_0b_3x3')(branch_2,
                                                is_training=is_training)
      with tf.variable_scope('Branch_3'):
        branch_3 = tf.nn.max_pool3d(net, ksize=[1, 3, 3, 3, 1],
                                    strides=[1, 1, 1, 1, 1], padding=snt.SAME,
                                    name='MaxPool3d_0a_3x3')
        branch_3 = Unit3D(output_channels=64, kernel_shape=[1, 1, 1],
                          name='Conv3d_0b_1x1')(branch_3,
                                                is_training=is_training)
      net = tf.concat([branch_0, branch_1, branch_2, branch_3], 4)
    end_points[end_point] = net
    if self._final_endpoint == end_point: return net, end_points

    end_point = 'Mixed_4c'
    with tf.variable_scope(end_point):
      with tf.variable_scope('Branch_0'):
        branch_0 = Unit3D(output_channels=160, kernel_shape=[1, 1, 1],
                          name='Conv3d_0a_1x1')(net, is_training=is_training)
      with tf.variable_scope('Branch_1'):
        branch_1 = Unit3D(output_channels=112, kernel_shape=[1, 1, 1],
                          name='Conv3d_0a_1x1')(net, is_training=is_training)
        branch_1 = Unit3D(output_channels=224, kernel_shape=[3, 3, 3],
                          name='Conv3d_0b_3x3')(branch_1,
                                                is_training=is_training)
      with tf.variable_scope('Branch_2'):
        branch_2 = Unit3D(output_channels=24, kernel_shape=[1, 1, 1],
                          name='Conv3d_0a_1x1')(net, is_training=is_training)
        branch_2 = Unit3D(output_channels=64, kernel_shape=[3, 3, 3],
                          name='Conv3d_0b_3x3')(branch_2,
                                                is_training=is_training)
      with tf.variable_scope('Branch_3'):
        branch_3 = tf.nn.max_pool3d(net, ksize=[1, 3, 3, 3, 1],
                                    strides=[1, 1, 1, 1, 1], padding=snt.SAME,
                                    name='MaxPool3d_0a_3x3')
        branch_3 = Unit3D(output_channels=64, kernel_shape=[1, 1, 1],
                          name='Conv3d_0b_1x1')(branch_3,
                                                is_training=is_training)
      net = tf.concat([branch_0, branch_1, branch_2, branch_3], 4)
    end_points[end_point] = net
    if self._final_endpoint == end_point: return net, end_points

    end_point = 'Mixed_4d'
    with tf.variable_scope(end_point):
      with tf.variable_scope('Branch_0'):
        branch_0 = Unit3D(output_channels=128, kernel_shape=[1, 1, 1],
                          name='Conv3d_0a_1x1')(net, is_training=is_training)
      with tf.variable_scope('Branch_1'):
        branch_1 = Unit3D(output_channels=128, kernel_shape=[1, 1, 1],
                          name='Conv3d_0a_1x1')(net, is_training=is_training)
        branch_1 = Unit3D(output_channels=256, kernel_shape=[3, 3, 3],
                          name='Conv3d_0b_3x3')(branch_1,
                                                is_training=is_training)
      with tf.variable_scope('Branch_2'):
        branch_2 = Unit3D(output_channels=24, kernel_shape=[1, 1, 1],
                          name='Conv3d_0a_1x1')(net, is_training=is_training)
        branch_2 = Unit3D(output_channels=64, kernel_shape=[3, 3, 3],
                          name='Conv3d_0b_3x3')(branch_2,
                                                is_training=is_training)
      with tf.variable_scope('Branch_3'):
        branch_3 = tf.nn.max_pool3d(net, ksize=[1, 3, 3, 3, 1],
                                    strides=[1, 1, 1, 1, 1], padding=snt.SAME,
                                    name='MaxPool3d_0a_3x3')
        branch_3 = Unit3D(output_channels=64, kernel_shape=[1, 1, 1],
                          name='Conv3d_0b_1x1')(branch_3,
                                                is_training=is_training)
      net = tf.concat([branch_0, branch_1, branch_2, branch_3], 4)
    end_points[end_point] = net
    if self._final_endpoint == end_point: return net, end_points

    end_point = 'Mixed_4e'
    with tf.variable_scope(end_point):
      with tf.variable_scope('Branch_0'):
        branch_0 = Unit3D(output_channels=112, kernel_shape=[1, 1, 1],
                          name='Conv3d_0a_1x1')(net, is_training=is_training)
      with tf.variable_scope('Branch_1'):
        branch_1 = Unit3D(output_channels=144, kernel_shape=[1, 1, 1],
                          name='Conv3d_0a_1x1')(net, is_training=is_training)
        branch_1 = Unit3D(output_channels=288, kernel_shape=[3, 3, 3],
                          name='Conv3d_0b_3x3')(branch_1,
                                                is_training=is_training)
      with tf.variable_scope('Branch_2'):
        branch_2 = Unit3D(output_channels=32, kernel_shape=[1, 1, 1],
                          name='Conv3d_0a_1x1')(net, is_training=is_training)
        branch_2 = Unit3D(output_channels=64, kernel_shape=[3, 3, 3],
                          name='Conv3d_0b_3x3')(branch_2,
                                                is_training=is_training)
      with tf.variable_scope('Branch_3'):
        branch_3 = tf.nn.max_pool3d(net, ksize=[1, 3, 3, 3, 1],
                                    strides=[1, 1, 1, 1, 1], padding=snt.SAME,
                                    name='MaxPool3d_0a_3x3')
        branch_3 = Unit3D(output_channels=64, kernel_shape=[1, 1, 1],
                          name='Conv3d_0b_1x1')(branch_3,
                                                is_training=is_training)
      net = tf.concat([branch_0, branch_1, branch_2, branch_3], 4)
    end_points[end_point] = net
    if self._final_endpoint == end_point: return net, end_points

    end_point = 'Mixed_4f'
    with tf.variable_scope(end_point):
      with tf.variable_scope('Branch_0'):
        branch_0 = Unit3D(output_channels=256, kernel_shape=[1, 1, 1],
                          name='Conv3d_0a_1x1')(net, is_training=is_training)
      with tf.variable_scope('Branch_1'):
        branch_1 = Unit3D(output_channels=160, kernel_shape=[1, 1, 1],
                          name='Conv3d_0a_1x1')(net, is_training=is_training)
        branch_1 = Unit3D(output_channels=320, kernel_shape=[3, 3, 3],
                          name='Conv3d_0b_3x3')(branch_1,
                                                is_training=is_training)
      with tf.variable_scope('Branch_2'):
        branch_2 = Unit3D(output_channels=32, kernel_shape=[1, 1, 1],
                          name='Conv3d_0a_1x1')(net, is_training=is_training)
        branch_2 = Unit3D(output_channels=128, kernel_shape=[3, 3, 3],
                          name='Conv3d_0b_3x3')(branch_2,
                                                is_training=is_training)
      with tf.variable_scope('Branch_3'):
        branch_3 = tf.nn.max_pool3d(net, ksize=[1, 3, 3, 3, 1],
                                    strides=[1, 1, 1, 1, 1], padding=snt.SAME,
                                    name='MaxPool3d_0a_3x3')
        branch_3 = Unit3D(output_channels=128, kernel_shape=[1, 1, 1],
                          name='Conv3d_0b_1x1')(branch_3,
                                                is_training=is_training)
      net = tf.concat([branch_0, branch_1, branch_2, branch_3], 4)
    end_points[end_point] = net
    if self._final_endpoint == end_point: return net, end_points

    end_point = 'MaxPool3d_5a_2x2'
    net = tf.nn.max_pool3d(net, ksize=[1, 2, 2, 2, 1], strides=[1, 2, 2, 2, 1],
                           padding=snt.SAME, name=end_point)
    end_points[end_point] = net
    if self._final_endpoint == end_point: return net, end_points

    end_point = 'Mixed_5b'
    with tf.variable_scope(end_point):
      with tf.variable_scope('Branch_0'):
        branch_0 = Unit3D(output_channels=256, kernel_shape=[1, 1, 1],
                          name='Conv3d_0a_1x1')(net, is_training=is_training)
      with tf.variable_scope('Branch_1'):
        branch_1 = Unit3D(output_channels=160, kernel_shape=[1, 1, 1],
                          name='Conv3d_0a_1x1')(net, is_training=is_training)
        branch_1 = Unit3D(output_channels=320, kernel_shape=[3, 3, 3],
                          name='Conv3d_0b_3x3')(branch_1,
                                                is_training=is_training)
      with tf.variable_scope('Branch_2'):
        branch_2 = Unit3D(output_channels=32, kernel_shape=[1, 1, 1],
                          name='Conv3d_0a_1x1')(net, is_training=is_training)
        branch_2 = Unit3D(output_channels=128, kernel_shape=[3, 3, 3],
                          name='Conv3d_0a_3x3')(branch_2,
                                                is_training=is_training)
      with tf.variable_scope('Branch_3'):
        branch_3 = tf.nn.max_pool3d(net, ksize=[1, 3, 3, 3, 1],
                                    strides=[1, 1, 1, 1, 1], padding=snt.SAME,
                                    name='MaxPool3d_0a_3x3')
        branch_3 = Unit3D(output_channels=128, kernel_shape=[1, 1, 1],
                          name='Conv3d_0b_1x1')(branch_3,
                                                is_training=is_training)
      net = tf.concat([branch_0, branch_1, branch_2, branch_3], 4)
    end_points[end_point] = net
    if self._final_endpoint == end_point: return net, end_points

    end_point = 'Mixed_5c'
    with tf.variable_scope(end_point):
      with tf.variable_scope('Branch_0'):
        branch_0 = Unit3D(output_channels=384, kernel_shape=[1, 1, 1],
                          name='Conv3d_0a_1x1')(net, is_training=is_training)
      with tf.variable_scope('Branch_1'):
        branch_1 = Unit3D(output_channels=192, kernel_shape=[1, 1, 1],
                          name='Conv3d_0a_1x1')(net, is_training=is_training)
        branch_1 = Unit3D(output_channels=384, kernel_shape=[3, 3, 3],
                          name='Conv3d_0b_3x3')(branch_1,
                                                is_training=is_training)
      with tf.variable_scope('Branch_2'):
        branch_2 = Unit3D(output_channels=48, kernel_shape=[1, 1, 1],
                          name='Conv3d_0a_1x1')(net, is_training=is_training)
        branch_2 = Unit3D(output_channels=128, kernel_shape=[3, 3, 3],
                          name='Conv3d_0b_3x3')(branch_2,
                                                is_training=is_training)
      with tf.variable_scope('Branch_3'):
        branch_3 = tf.nn.max_pool3d(net, ksize=[1, 3, 3, 3, 1],
                                    strides=[1, 1, 1, 1, 1], padding=snt.SAME,
                                    name='MaxPool3d_0a_3x3')
        branch_3 = Unit3D(output_channels=128, kernel_shape=[1, 1, 1],
                          name='Conv3d_0b_1x1')(branch_3,
                                                is_training=is_training)
      net = tf.concat([branch_0, branch_1, branch_2, branch_3], 4)
    end_points[end_point] = net
    if self._final_endpoint == end_point: return net, end_points

    end_point = 'Logits'
    with tf.variable_scope(end_point):
      net = tf.nn.avg_pool3d(net, ksize=[1, 2, 7, 7, 1],
                             strides=[1, 1, 1, 1, 1], padding=snt.VALID)
      net = tf.nn.dropout(net, dropout_keep_prob)
      logits = Unit3D(output_channels=self._num_classes,
                      kernel_shape=[1, 1, 1],
                      activation_fn=None,
                      use_batch_norm=False,
                      use_bias=True,
                      name='Conv3d_0c_1x1')(net, is_training=is_training)
      if self._spatial_squeeze:
        logits = tf.squeeze(logits, [2, 3], name='SpatialSqueeze')
    averaged_logits = tf.reduce_mean(logits, axis=1)
    end_points[end_point] = averaged_logits
    if self._final_endpoint == end_point: return averaged_logits, end_points

    end_point = 'Predictions'
    predictions = tf.nn.softmax(averaged_logits)
    end_points[end_point] = predictions
    return predictions, end_points


def check_chainer_activations_against_tf(model_chainer, chainer_dtype, input_dtype):
    # Loads both the original Tensorflow and Chainer model, checks that after
    # loading the checkpoint for both the model activations are the same for
    # each layer.
    tf_end_point_to_chainer = {
        "Conv3d_1a_7x7": "conv3d_1a_7x7",
        "MaxPool3d_2a_3x3": "max_pool_3d_2a_3x3",
        "Conv3d_2b_1x1": "conv3d_2b_1x1",
        "Conv3d_2c_3x3": "conv3d_2c_3x3",
        "MaxPool3d_3a_3x3": "max_pool_3d_3a_3x3",
        "Mixed_3b": "mixed_3b",
        "Mixed_3c": "mixed_3c",
        "MaxPool3d_4a_3x3": "max_pool_3d_4a_3x3",
        "Mixed_4b": "mixed_4b",
        "Mixed_4c": "mixed_4c",
        "Mixed_4d": "mixed_4d",
        "Mixed_4e": "mixed_4e",
        "Mixed_4f": "mixed_4f",
        "MaxPool3d_5a_2x2": "max_pool_3d_5a_2x2",
        "Mixed_5b": "mixed_5b",
        "Mixed_5c": "mixed_5c",
        "Logits": "averaged_logits",
    }
    input_tf = np.random.RandomState(8929042).uniform(
        low=-1, high=1, size=(1, 32, 224, 224, 3)).astype(np.float32)
    rgb_input_tf = tf.placeholder(tf.float32, shape=(1, None, None, None, 3))
    model_tf = I3D(num_classes=600, spatial_squeeze=True,
                   final_endpoint='Logits')
    _, end_points = model_tf(rgb_input_tf, is_training=False,
                             dropout_keep_prob=1.0)
    rgb_variable_map = {}
    for variable in tf.global_variables():
        rgb_variable_map[variable.name.replace(':0', '')[
                         len('inception_i3d/'):]] = variable
    saver = tf.train.Saver(reshape=True, var_list=rgb_variable_map)
    checkpoint_filename = _TEST_CHECKPOINT

    print("Computing model outputs with tensorflow...")
    with tf.Session() as sess:
        saver.restore(sess, checkpoint_filename)
        end_points_tf = sess.run(end_points, feed_dict={
            rgb_input_tf: input_tf
        })
    print("Computing model outputs with Chainer...")
    chainer.cuda.get_device_from_id(-1).use()
    with chainer.using_config('train', False):
        with chainer.using_config('dtype', chainer_dtype):
            print(chainer.config.dtype)
            input_chainer = cp.array(np.moveaxis(input_tf, 4, 1).astype(
                input_dtype))
            end_points_chainer = model_chainer(
                input_chainer, layers=list(tf_end_point_to_chainer.values()))
    print("Comparing results...")
    if chainer_dtype == np.float32:
        rtol = 1E-4
        atol = 1E-4
    else:
        rtol = 1E-1
        atol = 1E-1
    for tf_key, chainer_key in tf_end_point_to_chainer.items():
        print("Comparing {} and {}".format(tf_key, chainer_key))
        chainer_out = cp.asnumpy(end_points_chainer[chainer_key].data.astype(
            np.float32))
        tf_out = end_points_tf[tf_key]
        if chainer_out.ndim == 2:
            np.testing.assert_allclose(chainer_out, tf_out,
                                       rtol=rtol, atol=atol)
        else:
            np.testing.assert_allclose(np.moveaxis(chainer_out, 1, 4),
                                       tf_out, rtol=rtol, atol=atol)
    tf.reset_default_graph()


def test_tf_vs_chainer_i3d():
    for chainer_dtype, input_dtype in ((np.float32, np.float32),):
        model_chainer = i3d.I3D(num_classes=600, dropout_keep_prob=1.0)
        load_tensorflow_checkpoint(model_chainer, _TEST_CHECKPOINT)
        model_chainer.to_gpu(0)
        check_chainer_activations_against_tf(model_chainer, input_dtype, chainer_dtype)


def test_tf_vs_npz_i3d():
    with tempfile.NamedTemporaryFile() as tmp_npz_checkpoint_file:
        tf_checkpoint_to_npz(_TEST_CHECKPOINT, tmp_npz_checkpoint_file.name)
        for chainer_dtype, input_dtype in ((np.float32, np.float32),):
            model = i3d.I3D(600, 1.0)
            model.to_gpu(0)
            chainer.serializers.load_npz(tmp_npz_checkpoint_file.name, model)
            check_chainer_activations_against_tf(model, input_dtype, chainer_dtype)
