import numpy as np
import pytest

from .. import SlidingWindowSegmenter


@pytest.mark.parametrize(
    'multilingual_test_inputs, window_size, step_size',
    [
        ('It is a sunny day!!!! When Andy comes back, we are going to the zoo.', 20, 10),
        ('It is a sunny day!!!! When Andy comes back, we are going to the zoo.', 30, 10),
        ('It is a sunny day!!!! When Andy comes back, we are going to the zoo.', 30, 20),
        ('It is a sunny day!!!! When Andy comes back, we are going to the zoo.', 30, 29),
        ('ini adalah sebuah kalimat. ini adalah sebuah kalimat lain.', 5, 4),
        ('ini adalah sebuah kalimat. ini adalah sebuah kalimat lain.', 2, 1),
        ('ini adalah sebuah kalimat. ini adalah sebuah kalimat lain.', 200, 6),
        ('ini adalah sebuah kalimat. ini adalah sebuah kalimat lain.', 200, 1),
        ('今天是个大晴天！安迪回来以后，我们准备去动物园。', 1, 1),
        ('今天是个大晴天！安迪回来以后，我们准备去动物园。', 3, 2),
        ('Jina', 20, 10),
        ('J', 1, 1),
        ('J', 200, 1),
        ('J', 200, 2),
    ],
)
def test_sliding_window_segmenter_chunks(multilingual_test_inputs, window_size, step_size):
    segmenter = SlidingWindowSegmenter(window_size=window_size, step_size=step_size)

    text = multilingual_test_inputs
    expected_num_of_splits = 1 if len(text) < window_size else (max(0, len(text) - window_size) // step_size) + \
                                                               (min(1, (len(text) - window_size) % step_size) ^ 0) + 1

    docs_chunks = segmenter.segment(np.stack([text, text]))
    assert len(docs_chunks) == 2
    for chunks in docs_chunks:
        assert len(chunks) == expected_num_of_splits


@pytest.mark.parametrize(
    'multilingual_test_inputs, multilingual_test_expected_locations',
    [
        ('It is a sunny day!!!! When Andy comes back, we are going to the zoo.',
         [[0, 20], [10, 30], [20, 40], [30, 50], [40, 60], [50, 68]]),
        ('ini adalah sebuah kalimat. ini adalah sebuah kalimat lain.',
         [[0, 20], [10, 30], [20, 40], [30, 50], [40, 58]]),
        ('今天是个大晴天！安迪回来以后，我们准备去动物园。', [[0, 20], [10, 24]]),
        ('Jina', [[0, 4]]),
    ],
)
def test_sliding_window_segmenter_location(multilingual_test_inputs, multilingual_test_expected_locations):
    window_size = 20
    step_size = 10
    segmenter = SlidingWindowSegmenter(window_size=window_size, step_size=step_size)

    text = multilingual_test_inputs
    expected_locations = multilingual_test_expected_locations

    docs_chunks = segmenter.segment(np.stack([text, text]))
    assert len(docs_chunks) == 2
    for chunks in docs_chunks:
        for i, chunk in enumerate(chunks):
            assert chunk['location'] == expected_locations[i]
