import numpy as np

from .. import SlidingWindowAudioSlicer


def test_sliding_window_mono():
    n_frames = 100
    frame_length = 2048
    signal_orig = np.random.randn(frame_length * n_frames)

    crafter = SlidingWindowAudioSlicer(frame_length, frame_length // 2)
    crafted_chunks = crafter.craft(signal_orig, 0)

    assert len(crafted_chunks) == n_frames * 2 - 1


def test_sliding_window_stereo():
    n_frames = 100
    frame_length = 2048
    signal_orig = np.random.randn(2, frame_length * n_frames)

    crafter = SlidingWindowAudioSlicer(frame_length, frame_length // 2)
    crafted_chunks = crafter.craft(signal_orig, 0)

    assert len(crafted_chunks) == (n_frames * 2 - 1) * 2
