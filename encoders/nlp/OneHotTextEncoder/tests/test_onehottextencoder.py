from .. import OneHotTextEncoder
import numpy as np
import os
from jina.executors import BaseExecutor
import shutil

tmp_files = []
os.environ['TEST_WORKDIR'] = os.getcwd()

def teardown():
    print('tear down...')
    for k in tmp_files:
        if os.path.exists(k):
            if os.path.isfile(k):
                os.remove(k)
            elif os.path.isdir(k):
                shutil.rmtree(k, ignore_errors=False, onerror=None)

def add_tmpfile(*path):
    tmp_files.extend(path)

def test_encoding_results():
    encoder = OneHotTextEncoder(workspace=os.environ['TEST_WORKDIR'])
    test_data = np.array(['a', 'b', 'c', 'x', '!'])
    encoded_data = encoder.encode(test_data)
    assert encoded_data.shape == (5, 97)
    assert type(encoded_data) is np.ndarray


def test_save_and_load():
    encoder = OneHotTextEncoder(workspace=os.environ['TEST_WORKDIR'])
    encoder.save_config()
    assert os.path.exists(encoder.config_abspath)
    test_data = np.array(['a', 'b', 'c', 'x', '!'])
    encoded_data_control = encoder.encode(test_data)

    encoder.touch()
    encoder.save()
    assert os.path.exists(encoder.save_abspath)
    encoder_loaded = BaseExecutor.load(encoder.save_abspath)
    encoded_data_test = encoder_loaded.encode(test_data)

    np.testing.assert_array_equal(encoded_data_control, encoded_data_test)
    assert encoder_loaded.dim == encoder.dim
    add_tmpfile(
        encoder.config_abspath, encoder.save_abspath, encoder_loaded.config_abspath, encoder_loaded.save_abspath)


def test_save_and_load_config():
    encoder = OneHotTextEncoder(workspace=os.environ['TEST_WORKDIR'])
    encoder.save_config()
    assert os.path.exists(encoder.config_abspath)

    encoder_loaded = BaseExecutor.load_config(encoder.config_abspath)
    assert encoder_loaded.dim == encoder.dim
    add_tmpfile(encoder_loaded.config_abspath, encoder_loaded.save_abspath)