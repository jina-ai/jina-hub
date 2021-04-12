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
    frame_length = 10
    frame_overlap_length = frame_length // 2
    hop_length = frame_length - frame_overlap_length
    n_frames = 5
    num_docs = 3

    signal_orig = np.random.randn(frame_length * n_frames)
    expected_n_frames = (signal_orig.shape[0] - frame_length) / hop_length + 1
    expected_locations = [[i * hop_length, i * hop_length + frame_length] for i in range(int(expected_n_frames))]
    expected_channel = 'mono'

    segmenter = SlidingWindowAudioSlicer(frame_length=frame_length, frame_overlap_length=frame_overlap_length)
    docs = segmenter.segment(np.stack([signal_orig] * num_docs))

    assert len(docs) == num_docs
    for d in docs:
        assert len(d) == expected_n_frames
        for i, chunk in enumerate(d):
            assert chunk['location'] == expected_locations[int(i % expected_n_frames)]
            assert chunk['tags']['channel'] == expected_channel


def test_location_stereo():
    frame_length = 10
    frame_overlap_length = frame_length // 2
    hop_length = frame_length - frame_overlap_length
    n_frames = 5
    num_docs = 3
    num_channels = 2

    signal_orig = np.random.randn(num_channels, frame_length * n_frames)
    expected_n_frames = (signal_orig.shape[1] - frame_length) / hop_length + 1
    expected_locations = [[i * hop_length, i * hop_length + frame_length] for i in range(int(expected_n_frames))]

    segmenter = SlidingWindowAudioSlicer(frame_length=frame_length, frame_overlap_length=frame_overlap_length)
    docs = segmenter.segment(np.stack([signal_orig] * num_docs))

    assert len(docs) == num_docs
    for d in docs:
        assert len(d) == expected_n_frames * num_channels
        for i, chunk in enumerate(d):
            assert chunk['location'] == expected_locations[int(i % expected_n_frames)]
            expected_channel = 'left' if i // expected_n_frames == 0 else 'right'
            assert chunk['tags']['channel'] == expected_channel
