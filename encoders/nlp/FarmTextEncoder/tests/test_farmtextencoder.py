import os
import mock
import numpy as np
import shutil
import pytest

from .. import FarmTextEncoder
from jina.executors import BaseExecutor
from jina.executors.metas import get_default_metas

def get_metas():
    metas = get_default_metas()
    if 'JINA_TEST_GPU' in os.environ:
        metas['on_gpu'] = True
    return metas

def rm_files(tmp_files):
    for file in tmp_files:
        if file and os.path.exists(file):
            if os.path.isfile(file):
                os.remove(file)
            elif os.path.isdir(file):
                shutil.rmtree(file, ignore_errors=False, onerror=None)

class MockModule:
    def __init__(self):
        pass

    def extract_vectors(self, dicts, *args, **kwargs):
        print('i am a mocker function')
        for k in dicts:
            k['vec'] = np.array([i for i in range(target_output_dim)])
        return dicts

target_output_dim = 768
test_data = np.array(['it is a good day!', 'the dog sits on the floor.'])

def _test_encoding_results():
    metas = get_metas()
    encoder = FarmTextEncoder(metas=metas)
    encoded_data = encoder.encode(test_data)
    assert encoded_data.shape == (2, target_output_dim)



@pytest.mark.skipif('JINA_TEST_PRETRAINED' not in os.environ, reason='skip the pretrained test if not set')
def test_encoding_results():
    _test_encoding_results()

@mock.patch('farm.infer.Inferencer.load', return_value=MockModule())
def test_encoding_results_local(mocker):
    _test_encoding_results()

@mock.patch('farm.infer.Inferencer.load', return_value=MockModule())
def test_save_and_load(mocker):
    metas = get_metas()
    encoder = FarmTextEncoder(metas=metas)
    encoded_data_control = encoder.encode(test_data)
    encoder.touch()
    encoder.save()
    assert os.path.exists(encoder.save_abspath)
    encoder_loaded = BaseExecutor.load(encoder.save_abspath)
    encoded_data_test = encoder_loaded.encode(test_data)
    assert encoder_loaded.pretrained_model_name_or_path == encoder.pretrained_model_name_or_path
    np.testing.assert_array_equal(encoded_data_control, encoded_data_test)
    rm_files([encoder.save_abspath])

@mock.patch('farm.infer.Inferencer.load', return_value=MockModule())
def test_save_and_load_config(mocker):
    metas = get_metas()
    encoder = FarmTextEncoder(metas=metas)
    encoder.save_config()
    assert os.path.exists(encoder.config_abspath)
    encoder_loaded = BaseExecutor.load_config(encoder.config_abspath)
    assert encoder_loaded.pretrained_model_name_or_path == encoder.pretrained_model_name_or_path
    rm_files([encoder.config_abspath])
