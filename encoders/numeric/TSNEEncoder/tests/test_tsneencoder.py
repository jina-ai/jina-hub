__copyright__ = "Copyright (c) 2020 Jina AI Limited. All rights reserved."
__license__ = "Apache-2.0"

import os
import numpy as np
import shutil


from .. import TSNEEncoder
from jina.executors import BaseExecutor

requires_train_after_load = False
input_dim = 28
target_output_dim = 2

def rm_files(file_paths):
    for file_path in file_paths:
        if os.path.exists(file_path):
            if os.path.isfile(file_path):
                os.remove(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path, ignore_errors=False, onerror=None)

def test_encoder():
    encoder = TSNEEncoder(output_dim=target_output_dim)
    assert encoder is not None
    test_data = np.random.rand(10, input_dim)
    encoded_data = encoder.encode(test_data)
    assert encoded_data.shape == (test_data.shape[0], target_output_dim)
    assert type(encoded_data) is np.ndarray

def test_save_and_load():
    encoder = TSNEEncoder(output_dim=target_output_dim)
    assert encoder is not None
    test_data = np.random.rand(10, input_dim)
    encoded_data_control = encoder.encode(test_data)
    encoder.touch()
    encoder.save()
    assert os.path.exists(encoder.save_abspath)
    encoder_loaded = BaseExecutor.load(encoder.save_abspath)

    if not requires_train_after_load:
        # some models are not deterministic when training, so even with same training data, we cannot ensure
        # same encoding results
        encoded_data_test = encoder_loaded.encode(test_data)
        np.testing.assert_array_equal(
            encoded_data_test, encoded_data_control)
    rm_files([encoder.save_abspath])


def test_save_and_load_config():
    encoder = TSNEEncoder(output_dim=target_output_dim)
    assert encoder is not None
    encoder.save_config()
    encoder.touch()
    encoder.save()
    assert os.path.exists(encoder.save_abspath)
    encoder_loaded = BaseExecutor.load_config(encoder.config_abspath)
    test_data = np.random.rand(10, input_dim)
    encoded_data_test = encoder_loaded.encode(test_data)
    assert encoded_data_test.shape == (10, target_output_dim)
    rm_files([encoder.save_abspath, encoder.config_abspath])