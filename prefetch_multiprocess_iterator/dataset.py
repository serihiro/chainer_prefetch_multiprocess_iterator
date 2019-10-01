import os
import time

import numpy

try:
    from PIL import Image

    available = True
except ImportError as e:
    available = False
    _import_error = e
import bisect
import io
import six

import chainer
from chainer.datasets.image_dataset import LabeledImageDataset


def _read_image_as_array(path, dtype):
    f = Image.open(path)
    try:
        image = numpy.asarray(f, dtype=dtype)
    finally:
        # Only pillow >= 3.0 has 'close' method
        if hasattr(f, "close"):
            f.close()
    return image


def _postprocess_image(image):
    if image.ndim == 2:
        # image is greyscale
        image = image[..., None]
    return image.transpose(2, 0, 1)


class ExtendedLabeledImageDataset(LabeledImageDataset):
    def __init__(
        self, pairs, root=".", dtype=None, label_dtype=numpy.int32, measure=False
    ):
        _check_pillow_availability()
        if isinstance(pairs, six.string_types):
            pairs_path = pairs
            with open(pairs_path) as pairs_file:
                pairs = []
                for i, line in enumerate(pairs_file):
                    pair = line.strip().split()
                    if len(pair) != 2:
                        raise ValueError(
                            "invalid format at line {} in file {}".format(i, pairs_path)
                        )
                    pairs.append((pair[0], int(pair[1])))
        self._pairs = pairs
        self._root = root
        self._dtype = chainer.get_dtype(dtype)
        self._label_dtype = label_dtype
        self._measure = measure

    @property
    def pairs(self):
        return self._pairs

    @property
    def root(self):
        return self._root

    def get_example_by_path(self, full_path, int_label):
        if self._measure:  # timer
            get_example_timer = time.time()  # timer

        if self._measure:  # timer
            file_read_timer = time.time()  # timer
        f = Image.open(full_path)
        if self._measure:  # timer
            file_read_time = time.time() - file_read_timer  # timer
        try:
            image = numpy.asarray(f, dtype=self._dtype)
        finally:
            # Only pillow >= 3.0 has 'close' method
            if hasattr(f, "close"):
                f.close()

        label = numpy.array(int_label, dtype=self._label_dtype)
        if image.ndim == 2:
            # image is greyscale
            image = image[..., None]
        image = image.transpose(2, 0, 1)

        if self._measure:  # timer
            get_example_time = time.time() - get_example_timer  # timer
            return image, label, file_read_time, get_example_time  # timer
        else:
            return image, label


def _check_pillow_availability():
    if not available:
        raise ImportError(
            "PIL cannot be loaded. Install Pillow!\n"
            "The actual import error is as follows:\n" + str(_import_error)
        )
