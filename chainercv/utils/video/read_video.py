from itertools import count

import numpy as np
try:
    import cv2  # NOQA
    _cv2_available = True
except ImportError:
    _cv2_available = False


def read_video(file, start_frame=None, end_frame=None,
               dtype=np.float32, return_framerate=False):
    """ Read a video from a file.

    This function reads video frames from a given file. This returns an
    iterable of images where each image has the same shape and is in CHW
    format. The range of its values os :math:`[0, 255]`. The channels are in
    RGB order.

    This function requires "cv2" to be installed.

    :param file (string): The path of the file to open.
    :param start_frame (int or None): Start frame to start loading the video
    from. If it is `None`, start from the first video frame.
    :param end_frame (int or None): End frame to stop loading the video
    (exclusive). If it is `None`, load until the end of video.
    :param dtype: The type of the output arrays. The default value is
    :obj:`numpy.float32`
    :param return_framerate (boolean): Set to true to return the video's
    framerate in frames per second alongside the video.
    :return: An iterable of `numpy.ndarray`, each of which has shape (3, H, W).
    If both `start_frame` and `end_frame` are specified, this iterable is
    guaranteed to yield exactly `end_frame - start_frame` images. If
    `return_framerate` is True, the video framerate in frames per seconds
    is returned as a floating point value.
    """
    if not _cv2_available:
        raise ValueError('You need to have OpenCV installed to use '
                         'read_video.`')
    vc = cv2.VideoCapture(file)
    if not vc.isOpened():
        raise IOError("Could not open {}.".format(file))
    if start_frame is not None:
        vc.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    else:
        start_frame = 0

    frames = []

    if end_frame is None:
        frame_index_it = count(start=0, step=1)
    else:
        if end_frame <= start_frame:
            raise ValueError("end_frame should be greater then {}, but "
                             "is {}.".format(start_frame, end_frame))
        frame_index_it = range(end_frame - start_frame)

    for i in frame_index_it:
        retval, frame = vc.read()
        if not retval or frame is None:
            if end_frame is not None:
                raise ValueError("Could not read {} from frame {} to {}: "
                                 "could not read frame {}.".format(
                    file, start_frame, end_frame, start_frame + i))
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = frame.astype(dtype)
        frames.append(frame)
    frames = np.stack(frames, axis=0)
    # FHWC -> FCHW
    frames = np.moveaxis(frames, 3, 1)

    if not return_framerate:
        return frames
    else:
        return frames, vc.get(cv2.CAP_PROP_FPS)
