import numpy as np

from .. import AudioSlicer


def test_segment_mono():
    n_frames = 100
    frame_length = 2048
    signal_orig = np.random.randn(frame_length * n_frames)

    crafter = AudioSlicer(frame_length)
    crafted_chunks = crafter.craft(signal_orig, 0)

    assert len(crafted_chunks) == n_frames


def test_segment_stereo():
    n_frames = 100
    frame_length = 2048
    signal_orig = np.random.randn(2, frame_length * n_frames)

    crafter = AudioSlicer(frame_length)
    crafted_chunks = crafter.craft(signal_orig, 0)

    assert len(crafted_chunks) == n_frames * 2
