"""
Tests for I3D.
"""
import chainer

chainer.config.use_cudnn = 'always'
import numpy as np
import cupy as cp
import os
import tensorflow as tf
import tempfile
from collections import OrderedDict

import chainercv.links.model.i3d.i3d as i3d
from examples.i3d.tfckpt2npz import load_tensorflow_checkpoint, \
    tf_checkpoint_to_npz

_TEST_VIDEO_DIRECTORY = os.path.join(
    'data',
    'kinetics_val'
)
_KINETICS_LABEL_MAP = os.path.join(
    'models',
    'label_map.txt'
)
_TEST_CHECKPOINTS = OrderedDict([
    ("checkpoints", [os.path.join('models', 'rgb_scratch_kin600', 'model.ckpt'),
                     os.path.join('models', 'rgb_imagenet', 'model.ckpt'),
                     os.path.join('models', 'rgb_scratch', 'model.ckpt'),
                     os.path.join('models', 'flow_imagenet', 'model.ckpt'),
                     os.path.join('models', 'flow_scratch', 'model.ckpt')]),
    ("num_classes", [600, 400, 400, 400, 400]),
    ("input_channels", [3, 3, 3, 2, 2]),
    ('tf_scope_prefix', ('', 'RGB/inception_i3d/', 'RGB/inception_i3d/', 'Flow/inception_i3d/',
                         'Flow/inception_i3d/'))
])


def _get_checkpoints():
    return zip(*list(_TEST_CHECKPOINTS.values()))


# Include the original Tensorflow I3D model for testing purposes only.


def check_chainer_activations_against_tf(model_chainer, input_channels, num_classes, checkpoint_filename,
                                         chainer_dtype, input_dtype, scope_prefix):
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
        low=-1, high=1, size=(1, 32, 224, 224, input_channels)).astype(np.float32)
    rgb_input_tf = tf.placeholder(tf.float32, shape=(1, None, None, None, input_channels))
    model_tf = I3D(num_classes=num_classes, spatial_squeeze=True,
                   final_endpoint='Logits')
    _, end_points = model_tf(rgb_input_tf, is_training=False,
                             dropout_keep_prob=1.0)
    variable_map = {}
    for variable in tf.global_variables():
        variable_map[variable.name.replace(':0', '').replace('inception_i3d/', scope_prefix)] = variable
    saver = tf.train.Saver(reshape=True, var_list=variable_map)

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
            model_chainer.pick = list(tf_end_point_to_chainer.values())
            end_points_chainer = model_chainer(input_chainer)
            end_points_chainer = {
                key: val for key, val in
                zip(model_chainer.pick, end_points_chainer)
            }
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
        for checkpoint, num_classes, input_channels, scope_prefix in _get_checkpoints():
            model_chainer = i3d.I3D(num_classes=num_classes,
                                    dropout_keep_prob=1.0)
            load_tensorflow_checkpoint(model_chainer, checkpoint)
            model_chainer.to_gpu(0)
            check_chainer_activations_against_tf(model_chainer, input_channels, num_classes, checkpoint,
                                                 input_dtype, chainer_dtype, scope_prefix)


def test_tf_vs_npz_i3d():
    for checkpoint, num_classes, input_channels, scope_prefix in _get_checkpoints():
        with tempfile.NamedTemporaryFile() as tmp_npz_checkpoint_file:
            tf_checkpoint_to_npz(checkpoint, tmp_npz_checkpoint_file.name)
            tf.reset_default_graph()
            for chainer_dtype, input_dtype in ((np.float32, np.float32),):
                model = i3d.I3D(num_classes, 1.0)
                model.to_gpu(0)
                chainer.serializers.load_npz(tmp_npz_checkpoint_file.name, model)
                check_chainer_activations_against_tf(model, input_channels, num_classes, checkpoint,
                                                     input_dtype, chainer_dtype, scope_prefix)

