import numpy as np

from .. import SlidingWindowAudioSlicer


def test_sliding_window_mono():
    n_frames = 100
    frame_length = 2048
    signal_orig = np.random.randn(frame_length * n_frames)

    segmenter = SlidingWindowAudioSlicer(frame_length, frame_length // 2)
    segmented_chunks_per_doc = segmenter.segment(np.stack([signal_orig, signal_orig]))
    assert len(segmented_chunks_per_doc) == 2
    for segmented_chunk in segmented_chunks_per_doc:
        assert len(segmented_chunk) == n_frames * 2 - 1


def test_sliding_window_stereo():
    n_frames = 100
    frame_length = 2048
    signal_orig = np.random.randn(2, frame_length * n_frames)

    segmenter = SlidingWindowAudioSlicer(frame_length, frame_length // 2)
    segmented_chunks_per_doc = segmenter.segment(np.stack([signal_orig, signal_orig]))
    assert len(segmented_chunks_per_doc) == 2
    for segmented_chunk in segmented_chunks_per_doc:
        assert len(segmented_chunk) == (n_frames * 2 - 1) * 2
