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


def test_location_mono():
    n_frames = 5
    frame_length = 10
    signal_orig = np.random.randn(frame_length * n_frames)
    expected_locations = [[0, 10], [5, 15], [10, 20], [15, 25], [20, 30], [25, 35], [30, 40], [35, 45], [40, 50]]

    segmenter = SlidingWindowAudioSlicer(frame_length, frame_length // 2)

    segmented_chunks_per_doc = segmenter.segment(np.stack([signal_orig, signal_orig]))
    for segmented_chunk in segmented_chunks_per_doc:
        for i, chunk in enumerate(segmented_chunk):
            assert chunk['location'] == expected_locations[i]


def test_location_stereo():
    n_frames = 5
    frame_length = 10
    signal_orig = np.random.randn(2, frame_length * n_frames)
    expected_locations = [[0, 10], [5, 15], [10, 20], [15, 25], [20, 30], [25, 35], [30, 40], [35, 45], [40, 50]]

    segmenter = SlidingWindowAudioSlicer(frame_length, frame_length // 2)

    segmented_chunks_per_doc = segmenter.segment(np.stack([signal_orig, signal_orig]))
    for segmented_chunk in segmented_chunks_per_doc:
        for i, chunk in enumerate(segmented_chunk):
            assert chunk['location'] == expected_locations[i % (n_frames * 2 - 1)]
