import io
from typing import Dict

import numpy as np
from jina.executors.crafters import BaseCrafter


class ImageReader(BaseCrafter):
    """
    :class:`ImageReader` loads the image from the given file path, bytes or ndarray, and emits the unified `ndarray` with
    shape (h,w,depth).
    """

    def __init__(self, channel_axis: int = -1, *args, **kwargs):
        """
        :param channel_axis: the axis id of the color channel -1 indicates the color channel info at the last axis
        """
        super().__init__(*args, **kwargs)
        self.channel_axis = channel_axis

    def craft(self, buffer: bytes, uri: str, blob: 'np.ndarray', *args, **kwargs) -> Dict:
        """
        Read the image from the given `buffer`, `uri` or `blob` and saves the unified `ndarray` of the image in
        the `blob` of the document.

        :param buffer: the encoded image in raw bytes (png, jpg)
        :param uri: the image file path
        :param blob: the image as ndarray
        """

        pil_image = self.get_pil_image(buffer, uri, blob)
        pil_image = pil_image.convert('RGB')
        numpy_image = np.array(pil_image).astype('float32')
        if self.channel_axis != -1:
            numpy_image = np.moveaxis(numpy_image, -1, self.channel_axis)
        return dict(weight=1., blob=numpy_image)

    def validate_nd_image(self, nd_image):
        if nd_image.dtype != np.uint8:
            raise TypeError(f'ndarray has dtype {nd_image.dtype}, required {np.uint8}')
        shape_len = len(nd_image.shape)
        if not 2 <= shape_len <= 3:
            raise ValueError(f'image shape has x={shape_len} dimensions and violates 2 <= x <= 3')

    def get_pil_image(self, buffer, uri, blob):
        from PIL import Image
        if buffer:
            pil_image = Image.open(io.BytesIO(buffer))
        elif uri:
            pil_image = Image.open(uri)
        elif blob is not None:
            self.validate_nd_image(blob)
            pil_image = Image.fromarray(blob)
        else:
            raise ValueError('no value found in "buffer", "uri" and "blob"')
        return pil_image
