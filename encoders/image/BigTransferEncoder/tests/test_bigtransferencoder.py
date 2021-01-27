import os

import pytest
import numpy as np

from jina.executors.metas import get_default_metas
from jina.executors import BaseExecutor
from jina.excepts import PretrainedModelFileDoesNotExist

from .. import BigTransferEncoder


def get_encoder(model_path_tmp_dir, layer):
    metas = get_default_metas()
    if 'JINA_TEST_GPU' in os.environ:
        metas['on_gpu'] = True
    metas['workspace'] = model_path_tmp_dir
    return BigTransferEncoder(model_path='pretrained', channel_axis=1, layer=layer, metas=metas)

@pytest.mark.parametrize('layer', [None, 4])
def test_encoding_results(tmpdir, layer):
    input_dim = 48
    output_dim = 2048
    encoder = get_encoder(str(tmpdir), layer)
    test_data = np.random.rand(2, 3, input_dim, input_dim)
    encoded_data = encoder.encode(test_data)
    if layer is None:
        assert encoded_data.shape == (2, output_dim)
    else:
        assert encoded_data.shape == (2, 14, 14, 1024)

@pytest.mark.parametrize('layer', [None, 3])
def test_save_and_load(tmpdir, layer):
    encoder = get_encoder(str(tmpdir), layer)
    encoder.touch()
    encoder.save()
    assert os.path.exists(encoder.save_abspath)
    encoder_loaded = BaseExecutor.load(encoder.save_abspath)
    assert encoder_loaded.channel_axis == encoder.channel_axis


def test_raise_exception():
    with pytest.raises(PretrainedModelFileDoesNotExist):
        assert BigTransferEncoder(model_path=None)
