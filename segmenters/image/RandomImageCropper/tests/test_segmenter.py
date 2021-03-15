import numpy as np

from .. import RandomImageCropper


def create_random_img_array(img_height, img_width):
    return np.random.randint(0, 256, (img_height, img_width, 3))


def test_random_crop():
    img_size = 217
    img_array = create_random_img_array(img_size, img_size)
    output_dim = 20
    num_patches = 20
    segmenter = RandomImageCropper(output_dim, num_patches)
    docs_chunks = segmenter.segment(np.stack([img_array, img_array]))
    assert len(docs_chunks) == 2
    for chunks in docs_chunks:
        assert len(chunks) == num_patches
        for chunk in chunks:
            assert chunk['location'][0] <= (img_size - output_dim)
            assert chunk['location'][1] <= (img_size - output_dim)
