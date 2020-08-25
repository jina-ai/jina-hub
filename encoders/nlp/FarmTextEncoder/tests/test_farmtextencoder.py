import os
import pytest
import numpy as np

from .. import FarmTextEncoder
from jina.executors import BaseExecutor
from jina.executors.metas import get_default_metas

def get_metas():
    metas = get_default_metas()
    if 'JINA_TEST_GPU' in os.environ:
        metas['on_gpu'] = True
    return metas


# @pytest.mark.skipif('JINA_TEST_PRETRAINED' not in os.environ, reason='skip the pretrained test if not set')
def test_encoding_results():
    target_output_dim = 768 # why?
    metas = get_metas()
    encoder = FarmTextEncoder(metas=metas, max_length=10)
    test_data = np.array(['it is a good day!', 'the dog sits on the floor.'])
    encoded_data = encoder.encode(test_data)
    assert encoded_data.shape == (2, target_output_dim)

# @pytest.mark.skipif('JINA_TEST_PRETRAINED' not in os.environ, reason='skip the pretrained test if not set')
def test_save_and_load():
    metas = get_metas()
    encoder = FarmTextEncoder(metas=metas)
    test_data = np.array(['it is a good day!', 'the dog sits on the floor.'])
    encoded_data_control = encoder.encode(test_data)
    encoder.touch()
    encoder.save()
    assert os.path.exists(encoder.save_abspath)
    encoder_loaded = BaseExecutor.load(encoder.save_abspath)
    encoded_data_test = encoder_loaded.encode(test_data)
    assert encoder_loaded.max_length == encoder.max_length
    np.testing.assert_array_equal(encoded_data_control, encoded_data_test)

# @pytest.mark.skipif('JINA_TEST_PRETRAINED' not in os.environ, reason='skip the pretrained test if not set')
def test_save_and_load_config():
    metas = get_metas()
    encoder = FarmTextEncoder(metas=metas)
    encoder.save_config()
    assert os.path.exists(encoder.config_abspath)
    encoder_loaded = BaseExecutor.load_config(encoder.config_abspath)
    assert encoder_loaded.max_length == encoder.max_length
