import pytest
import numpy as np

from jina.excepts import PretrainedModelFileDoesNotExist
from jina.executors.metas import get_default_metas

from .. import Wav2VecSpeechEncoder


def test_encoding_results(tmpdir):
    target_output_dim = 512
    batch_size = 10
    signal_length = 1024
    test_data = np.random.randn(batch_size, signal_length).astype('f')
    metas = get_default_metas()
    metas['workspace'] = str(tmpdir)
    encoder = Wav2VecSpeechEncoder(model_path='/tmp/wav2vec_large.pt', input_sample_rate=16000, metas=metas)
    encoded_data = encoder.encode(test_data)
    assert encoded_data.shape[0] == batch_size
    assert encoded_data.shape[1] % target_output_dim == 0


def test_raise_exception():
    with pytest.raises(PretrainedModelFileDoesNotExist):
        Wav2VecSpeechEncoder()
