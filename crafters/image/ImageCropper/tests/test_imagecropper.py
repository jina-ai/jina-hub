__copyright__ = "Copyright (c) 2021 Jina AI Limited. All rights reserved."
__license__ = "Apache-2.0"

import os

import numpy as np
from PIL import Image

from .. import ImageCropper

cur_dir = os.path.dirname(os.path.abspath(__file__))


def create_random_img_array(img_height, img_width):
    return np.random.randint(0, 256, (img_height, img_width, 3))


def test_crop():
    img_size = 217
    img_array = create_random_img_array(img_size, img_size)
    left = 2
    top = 17
    width = 30
    height = 20
    crafter = ImageCropper(top=top, left=left, width=width, height=height)
    crafted_docs = crafter.craft(np.stack([img_array, img_array]))

    assert len(crafted_docs) == 2
    for crafted_doc in crafted_docs:
        np.testing.assert_array_equal(
            crafted_doc['blob'], np.asarray(img_array[top:top + height, left:left + width, :]),
            'img_array: {}\ntest: {}\ncontrol:{}'.format(
                img_array.shape,
                crafted_doc['blob'].shape,
                np.asarray(img_array[left:left + width, top:top + height, :]).shape))
        crop = Image.fromarray(np.uint8(crafted_doc['blob']))
        crop_width, crop_height = crop.size
        assert crop_width == width
        assert crop_height == height
        assert crafted_doc['location'] == (top, left)


def test_crop_file_image():
    tmp_fn = os.path.join(cur_dir, 'imgs/cars.jpg')
    img = Image.open(tmp_fn).convert('RGB')
    img_array = np.array(img).astype('float32')
    crafter = ImageCropper(top=541, left=992, width=24, height=67)
    crafted_docs = crafter.craft(np.stack([img_array, img_array]))
    assert len(crafted_docs) == 2
    for crafted_doc in crafted_docs:
        assert crafted_doc['blob'].shape == (67, 24, 3)
        crop_real_img = Image.open(os.path.join(cur_dir, 'imgs/faster_rcnn/person-0.png'))
        crop_real_img_array = np.array(crop_real_img).astype('float32')
        np.testing.assert_array_almost_equal(crafted_doc['blob'], crop_real_img_array)
