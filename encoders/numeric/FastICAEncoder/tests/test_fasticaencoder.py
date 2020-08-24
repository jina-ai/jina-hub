import numpy as np
import os
import pickle
import shutil

from jina.executors.encoders.numeric import TransformEncoder
from jina.executors import BaseExecutor
from .. import FastICAEncoder

input_dim = 28
target_output_dim = 2
train_data = np.random.rand(2000, input_dim)

def rm_files(tmp_files):
    for file in tmp_files:
        if file and os.path.exists(file):
            if os.path.isfile(file):
                os.remove(file)
            elif os.path.isdir(file):
                shutil.rmtree(file, ignore_errors=False, onerror=None)


def test_FastICATestCaseTrainCase():
    requires_train_after_load = True
    encoder = FastICAEncoder(
        output_dim=target_output_dim, whiten=True, num_features=input_dim, max_iter=200)
    encoder.train(train_data)
    encoding_results(encoder)
    save_and_load(encoder, requires_train_after_load)
    save_and_load_config(encoder, requires_train_after_load)
    rm_files([encoder.save_abspath, encoder.config_abspath, encoder.model_path])

def test_FastICATestCaseLoadCase():
    requires_train_after_load = False
    encoder = FastICAEncoder(
        output_dim=target_output_dim, whiten=True, num_features=input_dim, max_iter=200)
    encoder.train(train_data)
    filename = 'ica_model.model'
    pickle.dump(encoder.model, open(filename, 'wb'))
    encoder = TransformEncoder(model_path=filename)
    encoding_results(encoder)
    save_and_load(encoder, requires_train_after_load)
    save_and_load_config(encoder, requires_train_after_load)
    rm_files([encoder.save_abspath, encoder.config_abspath, encoder.model_path])


def encoding_results(encoder):
    test_data = np.random.rand(10, input_dim)
    encoded_data = encoder.encode(test_data)
    assert encoded_data.shape == (test_data.shape[0], target_output_dim)
    assert type(encoded_data) is np.ndarray



def save_and_load(encoder, requires_train_after_load):
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



def save_and_load_config(encoder, requires_train_after_load):
    test_data = np.random.rand(10, input_dim)
    encoder.save_config()
    assert os.path.exists(encoder.config_abspath)
    encoder_loaded = BaseExecutor.load_config(encoder.config_abspath)

    if requires_train_after_load:
        encoder_loaded.train(train_data)


    encoded_data_test = encoder_loaded.encode(test_data)
    assert encoded_data_test.shape == (10, target_output_dim)

