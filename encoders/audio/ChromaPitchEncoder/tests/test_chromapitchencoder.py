import numpy as np
from .. import ChromaPitchEncoder


def test_chroma_encoder():
    batch_size = 10
    n_frames = 5
    signal_length = 500 * n_frames
    test_data = np.random.randn(batch_size, signal_length)
    encoder = ChromaPitchEncoder()
    encoded_data = encoder.encode(test_data)
    assert encoded_data.shape == (batch_size, 12 * n_frames)