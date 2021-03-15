__copyright__ = "Copyright (c) 2021 Jina AI Limited. All rights reserved."
__license__ = "Apache-2.0"

import numpy as np
import pytest

from .. import ImageResizer


def create_random_img_array(img_height, img_width):
    return np.random.randint(0, 256, (img_height, img_width, 3))


def create_random_gray_img_array(img_height, img_width):
    return np.random.randint(0, 256, (img_height, img_width, 1))


def create_random_gray_img_array_2d(img_height, img_width):
    return np.random.randint(0, 256, (img_height, img_width))


def test_resize():
    img_width = 20
    img_height = 17

    # Test for int target_size
    output_dim = 71
    crafter = ImageResizer(target_size=output_dim)
    img_array = create_random_img_array(img_height, img_width)
    crafted_docs = crafter.craft(np.stack([img_array, img_array]))
    assert len(crafted_docs) == 2
    for crafted_doc in crafted_docs:
        assert min(crafted_doc['blob'].shape[:-1]) == output_dim

    # Test for tuple/list target_size
    output_dim = (img_height, img_width)
    crafter = ImageResizer(target_size=output_dim)
    img_array = create_random_img_array(img_width, img_height)
    crafted_docs = crafter.craft(np.stack([img_array, img_array]))
    assert len(crafted_docs) == 2
    for crafted_doc in crafted_docs:
        assert crafted_doc['blob'].shape[:-1] == output_dim


@pytest.mark.parametrize('img_array', [create_random_gray_img_array(17, 20),
                                       create_random_gray_img_array_2d(17, 20)]
                         )
def test_resize_gray(img_array):
    img_width = 20
    img_height = 17

    # Test for int target_size
    output_dim = 71
    crafter = ImageResizer(target_size=output_dim)
    crafted_docs = crafter.craft(np.stack([img_array, img_array]))
    assert len(crafted_docs) == 2
    for crafted_doc in crafted_docs:
        assert min(crafted_doc['blob'].shape[:-1]) == output_dim

    # Test for tuple/list target_size
    output_dim = (img_height, img_width)
    crafter = ImageResizer(target_size=output_dim)
    crafted_docs = crafter.craft(np.stack([img_array, img_array]))
    assert len(crafted_docs) == 2
    for crafted_doc in crafted_docs:
        assert crafted_doc['blob'].shape == output_dim
