import numpy as np

from .. import SlidingWindowSegmenter


def test_sliding_window_segmenter():
    window_size = 20
    step_size = 10
    segmenter = SlidingWindowSegmenter(
        window_size=window_size, step_size=step_size)
    text = 'It is a sunny day!!!! When Andy comes back, we are going to the zoo.'
    docs_chunks = segmenter.segment(np.stack([text, text]))
    assert len(docs_chunks) == 2
    for chunks in docs_chunks:
        assert len(chunks) == len(text) // step_size
