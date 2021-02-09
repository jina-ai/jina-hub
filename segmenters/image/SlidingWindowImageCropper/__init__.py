from typing import Tuple, Dict, List, Union

import numpy as np
from jina.executors.segmenters import BaseSegmenter

from .helper import _move_channel_axis


class SlidingWindowImageCropper(BaseSegmenter):
    """
    :class:`SlidingWindowImageCropper` crops the image with a sliding window.

    :param target_size: desired output size. If size is a sequence like (h, w),
        the output size will be matched to this.
        If size is an int, the output will have the
        same height and width as the `target_size`.
    :param strides: the strides between two neighboring sliding windows.
        `strides` is a sequence like (h, w),
        in which denote the strides on the vertical
        and the horizontal axis.
    :param padding: If False, only patches which are
        fully contained in the input image are included.
        If True, all patches whose starting point
        is inside the input are included,
        and areas outside the input default to zero.
        The `padding` argument has no effect on the size of each patch,
        it determines how many patches are extracted.
        Default is False.
    :param args:  Additional positional arguments
    :param kwargs: Additional keyword arguments
    """

    def __init__(self,
                 target_size: Union[Tuple[int], int] = 16,
                 strides: Tuple[int, int] = (2, 2),
                 padding: bool = False,
                 channel_axis: int = -1,
                 *args,
                 **kwargs):
        """Set constructor"""
        super().__init__(*args, **kwargs)
        self.target_size = target_size
        if len(strides) != 2:
            raise ValueError(f'strides should be a tuple of two integers: {strides}')
        self.stride_h, self.stride_w = strides
        self.padding = padding
        self.channel_axis = channel_axis

    def _add_zero_padding(self, img: 'np.ndarray') -> 'np.ndarray':
        h, w, c = img.shape
        ext_h = self.target_size - h % self.stride_h
        ext_w = self.target_size - w % self.stride_w
        return np.pad(img,
                      ((0, ext_h), (0, ext_w), (0, 0)),
                      mode='constant',
                      constant_values=0)

    def segment(self, blob: 'np.ndarray', *args, **kwargs) -> List[Dict]:
        """
        Crop the input image array with a sliding window.

        :param blob: the ndarray of the image with the color channel at the last axis
        :return: a list of chunk dicts with the cropped images.
        :param args:  Additional positional arguments
        :param kwargs: Additional keyword arguments
        """
        raw_img = np.copy(blob)
        raw_img = _move_channel_axis(raw_img, self.channel_axis)
        if self.padding:
            raw_img = self._add_zero_padding(blob)
        h, w, c = raw_img.shape
        row_step = raw_img.strides[0]
        col_step = raw_img.strides[1]

        expanded_img = np.lib.stride_tricks.as_strided(
            raw_img,
            shape=(
                1 + int((h - self.target_size) / self.stride_h),
                1 + int((w - self.target_size) / self.stride_w),
                self.target_size,
                self.target_size,
                c
            ),
            strides=(
                row_step * self.stride_h,
                col_step * self.stride_w,
                row_step,
                col_step,
                1), writeable=False)

        bbox_locations = [
            (h * self.stride_h, w * self.stride_w)
            for h in range(expanded_img.shape[0])
            for w in range(expanded_img.shape[1])]

        expanded_img = expanded_img.reshape((-1, self.target_size, self.target_size, c))

        results = []
        for location, _blob in zip(bbox_locations, expanded_img):
            blob = _move_channel_axis(_blob, -1, self.channel_axis)
            results.append(dict(offset=0, weight=1.0, blob=blob.astype('float32'), location=location))
        return results
