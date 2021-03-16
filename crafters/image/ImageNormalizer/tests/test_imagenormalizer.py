__copyright__ = "Copyright (c) 2021 Jina AI Limited. All rights reserved."
__license__ = "Apache-2.0"

import numpy as np

from .. import ImageNormalizer


def create_random_img_array(img_height, img_width):
    return np.random.randint(0, 256, (img_height, img_width, 3))


def test_transform_results():
    img_size = 217
    target_size = 224
    crafter = ImageNormalizer(target_size=target_size)
    img_array = create_random_img_array(img_size, img_size)
    crafted_docs = crafter.craft(np.stack([img_array, img_array]))
    assert len(crafted_docs) == 2
    for crafted_doc in crafted_docs:
        assert crafted_doc["blob"].shape == (224, 224, 3)

    img_size = (217, 200)
    target_size = (100, 224)
    crafter = ImageNormalizer(target_size=target_size)
    img_array = create_random_img_array(*img_size)
    crafted_docs = crafter.craft(np.stack([img_array, img_array]))
    assert len(crafted_docs) == 2
    for crafted_doc in crafted_docs:
        assert crafted_doc["blob"].shape[:-1] == target_size
