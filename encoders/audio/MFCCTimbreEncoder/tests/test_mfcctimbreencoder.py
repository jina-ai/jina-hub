from .. import MFCCTimbreEncoder
import numpy as np



def test_mfcc_encoder():
    batch_size = 10
    n_frames = 5
    signal_length = 500 * n_frames
    test_data = np.random.randn(batch_size, signal_length)
    n_mfcc = 12
    encoder = MFCCTimbreEncoder(n_mfcc=n_mfcc)
    encoded_data = encoder.encode(test_data)
    assert encoded_data.shape == (batch_size, n_mfcc * n_frames)
