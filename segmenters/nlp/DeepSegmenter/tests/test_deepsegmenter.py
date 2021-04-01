import numpy as np

from .. import DeepSegmenter


def test_deepsegmenter():
    segmenter = DeepSegmenter()
    text = 'I am Batman i live in gotham'
    docs_chunks = segmenter.segment(np.stack([text, text]))
    assert len(docs_chunks) == 2
    for chunks in docs_chunks:
        assert len(chunks) == 2
        assert chunks[0]['text'] == 'I am Batman'
        assert chunks[1]['text'] == 'i live in gotham'
