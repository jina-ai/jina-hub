from typing import Tuple, Dict, List, Union

import numpy as np

from jina.executors.decorators import single
from jina.executors.segmenters import BaseSegmenter

from .helper import _crop_image, _move_channel_axis, _load_image


class RandomImageCropper(BaseSegmenter):
    """
    :class:`RandomImageCropper` crops the image with a random crop box.
    The coordinate is the same coordinate-system that the :py:mode:`PIL.Image`.

    :param target_size: desired output size. If size is a sequence like (h, w), the output size will be matched to
        this. If size is an int, the output will have the same height and width as the `target_size`.
    :param num_patches: The number of crops to be done
    :param channel_axis: Axis for channel
    :param args:  Additional positional arguments
    :param kwargs: Additional keyword arguments
    """

    def __init__(self,
                 target_size: Union[Tuple[int], int] = 224,
                 num_patches: int = 1,
                 channel_axis: int = -1,
                 *args,
                 **kwargs):
        """Set constructor"""
        super().__init__(*args, **kwargs)
        self.target_size = target_size
        self.num_patches = num_patches
        self.channel_axis = channel_axis

    @single
    def segment(self, blob: 'np.ndarray', *args, **kwargs) -> List[Dict]:
        """
        Crop the input image array.

        :param blob: the ndarray of the image
        :return: a list of chunk dicts with the cropped images
        """
        raw_img = _load_image(blob, self.channel_axis)
        result = []
        for _ in range(self.num_patches):
            _img, top, left = _crop_image(raw_img, self.target_size, how='random')
            img = _move_channel_axis(np.asarray(_img), -1, self.channel_axis)
            result.append(
                dict(offset=0, weight=1., blob=np.asarray(img).astype('float32'), location=(top, left)))
        return result
