__copyright__ = "Copyright (c) 2021 Jina AI Limited. All rights reserved."
__license__ = "Apache-2.0"

from typing import Dict

import numpy as np

from jina.executors.decorators import single
from jina.executors.crafters import BaseCrafter

from .helper import _load_image, _move_channel_axis


class ImageFlipper(BaseCrafter):
    """
    Flip the image horizontally or vertically.

    Flip image in the left/right or up/down direction respectively.

    :param vertical: desired rotation type.
        ``True`` indicates the image should be flipped vertically.
    :param channel_axis: the axis id of the color channel, ``-1``
        indicates the color channel info at the last axis.
    :param args:  Additional positional arguments
    :param kwargs: Additional keyword arguments
    """

    def __init__(self,
                 vertical: bool = False,
                 channel_axis: int = -1,
                 *args,
                 **kwargs):
        """Set Constructor."""
        super().__init__(*args, **kwargs)
        self.vertical = vertical
        self.channel_axis = channel_axis

    @single
    def craft(self, blob: 'np.ndarray', *args, **kwargs) -> Dict:
        """
        Flip the input image array horizontally or vertically.

        :param blob: the ndarray of the image with the color channel at the last axis
        :return: A dict with the flipped image
        """
        raw_img = _load_image(blob, self.channel_axis)
        img = np.array(raw_img).astype('float32')
        img = np.flipud(img) if self.vertical else np.fliplr(img)
        img = _move_channel_axis(img, -1, self.channel_axis)
        return dict(offset=0, blob=img)
