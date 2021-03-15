__copyright__ = "Copyright (c) 2021 Jina AI Limited. All rights reserved."
__license__ = "Apache-2.0"

import numpy as np

from .. import ImageFlipper


def create_random_img_array(img_height, img_width):
    return np.random.randint(0, 256, (img_height, img_width, 3))


def test_horizontal_flip():
    img_size = 217
    crafter = ImageFlipper()
    # generates a random image array of size (217, 217)
    img_array = create_random_img_array(img_size, img_size)
    crafted_docs = crafter.craft(np.stack([img_array, img_array]))
    assert len(crafted_docs) == 2
    flip_img_array = np.fliplr(img_array)

    for crafted_doc in crafted_docs:
        # image flips along the second axis (horizontal flip)
        # assert flipped image using numpy's fliplr method
        np.testing.assert_equal(crafted_doc['blob'], flip_img_array)


def test_vertical_flip():
    img_size = 217
    crafter = ImageFlipper(vertical=True)
    # generates a random image array of size (217, 217)
    img_array = create_random_img_array(img_size, img_size)
    crafted_docs = crafter.craft(np.stack([img_array, img_array]))
    assert len(crafted_docs) == 2
    flip_img_array = np.flipud(img_array)

    for crafted_doc in crafted_docs:
        # image flips along the second axis (horizontal flip)
        # assert flipped image using numpy's fliplr method
        np.testing.assert_equal(crafted_doc['blob'], flip_img_array)
