import io
from typing import Dict

import numpy as np
from jina.executors.crafters import BaseCrafter


class ImageReader(BaseCrafter):
    """
    :class:`ImageReader` loads the image from the given file path and save the `ndarray` of the image in the Document.
    """

    def __init__(self, channel_axis: int = -1, *args, **kwargs):
        """
        :class:`ImageReader` load an image file and craft into image matrix.

        :param channel_axis: the axis id of the color channel, ``-1`` indicates the color channel info at the last axis
        """
        super().__init__(*args, **kwargs)
        self.channel_axis = channel_axis

    def craft(self, buffer: bytes, uri: str, *args, **kwargs) -> Dict:
        """
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
        # Add image metadata
        from PIL.ExifTags import TAGS
        from PIL.ExifTags import GPSTAGS 
        # Extract image metadata
        # https://github.com/python-pillow/Pillow/blob/master/src/PIL/ExifTags.py
        metadata = {}
        exif_data = raw_img.getexif()
        for tag_id in exif_data:
            tag = TAGS.get(tag_id)
            if tag is None:
                tag = GPSTAGS.get(tag_id, tag_id)
            data = exif_data.get(tag_id)
            # transform bytes to readable data
            if isinstance(data, bytes):
                data = data.decode()
            # if tag does not exists in Pillow
            if tag is None:
                continue
            metadata[tag] = data
        return dict(weight=1., blob=img, metadata=metadata)
