from .. import ImageNormalizer


def create_random_img_array(img_height, img_width):
    import numpy as np
    return np.random.randint(0, 256, (img_height, img_width, 3))


def test_transform_results():
    img_size = 217
    target_size = 224
    crafter = ImageNormalizer(target_size=target_size)
    img_array = create_random_img_array(img_size, img_size)
    crafted_doc = crafter.craft(img_array)
    assert crafted_doc["blob"].shape == (224, 224, 3)

    img_size = (217, 200)
    target_size = (100, 224)
    crafter = ImageNormalizer(target_size=target_size)
    img_array = create_random_img_array(*img_size)
    crafted_doc = crafter.craft(img_array)
    assert crafted_doc["blob"].shape[:-1] == target_size


def test_transform_std_greater_zero():
    # All img_std values must be greater than 0
    import pytest    
    with pytest.raises(ValueError):
        ImageNormalizer(img_std=(1, 0, 1))

    with pytest.raises(ValueError):
        ImageNormalizer(img_std=(1, 0.0, 1))

    with pytest.raises(ValueError):
        ImageNormalizer(img_std=(-1, 1.0, 1))
