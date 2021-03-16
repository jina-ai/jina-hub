__copyright__ = "Copyright (c) 2021 Jina AI Limited. All rights reserved."
__license__ = "Apache-2.0"

import numpy as np
from PIL import Image

from .. import CenterImageCropper


def create_random_img_array(img_height, img_width):
    return np.random.randint(0, 256, (img_height, img_width, 3))


def test_center_crop():
    img_size = 217
    img_array = create_random_img_array(img_size, img_size)
    width = 30
    height = 20
    output_dim = (height, width)
    crafter = CenterImageCropper(output_dim)
    crafted_docs = crafter.craft(np.stack([img_array, img_array]))
    assert len(crafted_docs) == 2
    for crafted_doc in crafted_docs:
        assert crafted_doc['blob'].shape == (height, width, 3)
        # int((img_size - output_dim) / 2)
        crop = Image.fromarray(np.uint8(crafted_doc['blob']))
        crop_width, crop_height = crop.size
        assert crop_width == width
        assert crop_height == height
        (top, left) = (98, 93)
        assert crafted_doc['location'] == (top, left)
