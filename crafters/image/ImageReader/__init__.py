import io
from typing import Dict

import numpy as np
from jina.executors.crafters import BaseCrafter


class ImageReader(BaseCrafter):
    """
    Load image file and craft it into image matrix.

    :class:`ImageReader` loads the image from the given file
        path and save the `ndarray` of the image in the Document.

    :param channel_axis: the axis id of the color channel.
        The ``-1`` indicates the color channel info at the last axis
    """

    def __init__(self, channel_axis: int = -1, *args, **kwargs):
        """Set Constructor."""
        super().__init__(*args, **kwargs)
        self.channel_axis = channel_axis

    def craft(self, buffer: bytes, uri: str, *args, **kwargs) -> Dict:
        """
        Read image file and craft it into image matrix.

        Read the image from the given file path that specified in `buffer` and save the `ndarray` of the image in
            the `blob` of the document.

        :param buffer: the image in raw bytes
        :param uri: the image file path

        """
        from PIL import Image
        if buffer:
            raw_img = Image.open(io.BytesIO(buffer))
        elif uri:
            raw_img = Image.open(uri)
        else:
            raise ValueError('no value found in "buffer" and "uri"')
        raw_img = raw_img.convert('RGB')
        img = np.array(raw_img).astype('float32')
        if self.channel_axis != -1:
            img = np.moveaxis(img, -1, self.channel_axis)
        return dict(blob=img)
