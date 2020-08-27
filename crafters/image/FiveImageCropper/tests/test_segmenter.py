from .. import FiveImageCropper


def create_random_img_array(img_height, img_width):
    import numpy as np
    return np.random.randint(0, 256, (img_height, img_width, 3))


def test_five_image_crop():
    img_size = 217
    img_array = create_random_img_array(img_size, img_size)
    output_dim = 20
    crafter = FiveImageCropper(output_dim)
    chunks = crafter.craft(img_array)
    assert len(chunks) == 5
    assert chunks[0]['location'] == (0, 0)
    assert chunks[1]['location'] == (0, 197)
    assert chunks[2]['location'] == (197, 0)
    assert chunks[3]['location'] == (197, 197)
    assert chunks[4]['location'] == (98, 98)
