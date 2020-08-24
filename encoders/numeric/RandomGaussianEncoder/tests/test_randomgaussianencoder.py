import os
import pickle
import shutil

import numpy as np

from .. import RandomGaussianEncoder
from jina.executors import BaseExecutor
from ..transformerencoder import TransformEncoder

input_dim = 28
target_output_dim = 2


def rm_files(file_paths):
    for file_path in file_paths:
        if os.path.exists(file_path):
            if os.path.isfile(file_path):
                os.remove(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path, ignore_errors=False, onerror=None)


def test_randomgaussianencodertrain():
    train_data = np.random.rand(2000, input_dim)
    requires_train_after_load = True
    encoder = RandomGaussianEncoder(output_dim=target_output_dim)
    encoder.train(train_data)
    encoding_results(encoder)
    transform_encoder = None
    save_and_load(encoder, transform_encoder, requires_train_after_load)
    save_and_load_config(encoder, transform_encoder, requires_train_after_load, train_data)
    rm_files([encoder.save_abspath])


def test_randomgaussianencoderload():
    train_data = np.random.rand(2000, input_dim)
    requires_train_after_load = False
    encoder = RandomGaussianEncoder(output_dim=target_output_dim)
    filename = 'random_gaussian_model.model'
    pickle.dump(encoder.model.fit_transform(train_data), open(filename, 'wb'))
    transform_encoder = TransformEncoder(model_path=filename)
    save_and_load(encoder, transform_encoder, requires_train_after_load)
    save_and_load_config(encoder, transform_encoder, requires_train_after_load, train_data)
    rm_files([transform_encoder.save_abspath])
    rm_files([filename])
    rm_files([encoder.save_abspath])

def encoding_results(encoder):
    assert encoder is not None
    test_data = np.random.rand(10, input_dim)
    encoded_data = encoder.encode(test_data)
    assert encoded_data.shape == (test_data.shape[0], target_output_dim)
    assert type(encoded_data) == np.ndarray


def save_and_load(encoder, transform_encoder, requires_train_after_load):
    assert encoder is not None
    test_data = np.random.rand(10, input_dim)

    encoded_data_control = encoder.encode(test_data)
    encoder.touch()
    encoder.save()
    assert os.path.exists(encoder.save_abspath)

    if not requires_train_after_load:
        transform_encoder.touch()
        transform_encoder.save()
        assert os.path.exists(transform_encoder.save_abspath)
        encoder_loaded = BaseExecutor.load(transform_encoder.save_abspath)
        # some models are not deterministic when training, so even with same training data, we cannot ensure
        # same encoding results
        encoded_data_test = encoder.encode(test_data)
        np.testing.assert_array_equal(
            encoded_data_test, encoded_data_control)



def save_and_load_config(encoder, transform_encoder, requires_train_after_load, train_data):
    assert encoder is not None

    test_data = np.random.rand(10, input_dim)

    encoder.save_config()
    assert os.path.exists(encoder.save_abspath)
    encoder_loaded = BaseExecutor.load_config(encoder.config_abspath)

    if requires_train_after_load:
        encoder_loaded = BaseExecutor.load_config(encoder.config_abspath)
        encoder_loaded.train(train_data)
        encoded_data_test = encoder_loaded.encode(test_data)
        assert encoded_data_test.shape == (10, target_output_dim)

    if not requires_train_after_load:
        transform_encoder.save_config()
        assert os.path.exists(transform_encoder.save_abspath)


