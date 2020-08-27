import os
import pytest
import numpy as np

from .. import Wav2VecSpeechEncoder


@pytest.mark.skipif('JINA_TEST_PRETRAINED' not in os.environ, reason='skip the pretrained test if not set')
def test_encoding_results():
    target_output_dim = 512
    batch_size = 10
    signal_length = 1024
    test_data = np.random.randn(batch_size, signal_length).astype('f')
    encoder = Wav2VecSpeechEncoder(model_path='/tmp/wav2vec_large.pt', input_sample_rate=16000)
    encoded_data = encoder.encode(test_data)
    assert encoded_data.shape[0] == batch_size
    assert encoded_data.shape[1] % target_output_dim == 0
