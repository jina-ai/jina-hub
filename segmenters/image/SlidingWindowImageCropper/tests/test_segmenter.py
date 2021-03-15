import numpy as np

from .. import SlidingWindowImageCropper


def create_random_img_array(img_height, img_width):
    return np.random.randint(0, 256, (img_height, img_width, 3))


def test_sliding_windows_no_padding():
    img_size = 14
    img_array = create_random_img_array(img_size, img_size)
    output_dim = 4
    strides = (6, 6)
    segmenter = SlidingWindowImageCropper(target_size=output_dim, strides=strides, padding=False)
    docs_chunks = segmenter.segment(np.stack([img_array, img_array]))
    assert len(docs_chunks) == 2
    for chunks in docs_chunks:
        assert len(chunks) == 4
        assert chunks[0]['location'] == (0, 0)
        assert chunks[1]['location'] == (0, 6)
        assert chunks[2]['location'] == (6, 0)
        assert chunks[3]['location'] == (6, 6)


def test_sliding_windows_with_padding():
    img_size = 14
    img_array = create_random_img_array(img_size, img_size)
    output_dim = 4
    strides = (6, 6)
    segmenter = SlidingWindowImageCropper(target_size=output_dim, strides=strides, padding=True)
    docs_chunks = segmenter.segment(np.stack([img_array, img_array]))
    assert len(docs_chunks) == 2
    for chunks in docs_chunks:
        assert len(chunks) == 9
        assert chunks[0]['location'] == (0, 0)
        assert chunks[1]['location'] == (0, 6)
        assert chunks[2]['location'] == (0, 12)
        assert chunks[3]['location'] == (6, 0)
        assert chunks[4]['location'] == (6, 6)
        assert chunks[5]['location'] == (6, 12)
        assert chunks[6]['location'] == (12, 0)
        assert chunks[7]['location'] == (12, 6)
        assert chunks[8]['location'] == (12, 12)


def test_sliding_windows_without_padding_rectangular_ugly_shapes():
    height = 16
    width = 11
    img_array = create_random_img_array(img_height=height, img_width=width)
    output_dim = 4
    strides = (4, 4)
    segmenter = SlidingWindowImageCropper(target_size=output_dim, strides=strides, padding=False)
    docs_chunks = segmenter.segment(np.stack([img_array, img_array]))
    assert len(docs_chunks) == 2
    for chunks in docs_chunks:
        assert len(chunks) == 8
        assert chunks[0]['location'] == (0, 0)
        assert chunks[1]['location'] == (0, 4)
        assert chunks[2]['location'] == (4, 0)
        assert chunks[3]['location'] == (4, 4)
        assert chunks[4]['location'] == (8, 0)
        assert chunks[5]['location'] == (8, 4)
        assert chunks[6]['location'] == (12, 0)
        assert chunks[7]['location'] == (12, 4)
