from .. import CompressionVaeEncoder

import os
import shutil
import numpy as np
import pytest

from cvae import cvae

from jina.executors import BaseExecutor


def rm_files(file_paths):
    for file_path in file_paths:
        if os.path.exists(file_path):
            if os.path.isfile(file_path):
                os.remove(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path, ignore_errors=False, onerror=None)


@pytest.fixture(scope="function")
def test_data():
    """
    Create an array of the given shape and populate it with random samples from a uniform distribution over [0, 1).
    :return: a `B x T` numpy ``ndarray``, `B` is the size of the batch
    """
    batch_size = 10
    input_dim = 28
    test_data = np.random.rand(batch_size, input_dim)

    return test_data


@pytest.fixture(scope="function")
def target_output_dim():
    """
    The size of the latent space.
    """
    target_output_dim = 2

    return target_output_dim


@pytest.fixture(scope="function")
def get_encoder(tmpdir, test_data, target_output_dim):
    model_path = str(tmpdir.mkdir('model').join('model'))
    data_path = str(tmpdir.mkdir('data'))

    for idx, features in enumerate(test_data):
        np.save(os.path.join(data_path, str(idx)), features)

    # Train the CVAE on the test data to build a model saved in `logdir`.
    model = cvae.CompressionVAE(
        data_path, dim_latent=target_output_dim, logdir=model_path
    )
    model.train()

    return CompressionVaeEncoder(
        model_path=model_path, X=data_path, output_dim=target_output_dim
    )


def test_encoding_results(get_encoder, test_data, target_output_dim):
    expected_batch_size = test_data.shape[0]

    encoder = get_encoder
    assert encoder is not None

    encoded_data = encoder.encode(test_data)
    assert encoded_data.shape == (expected_batch_size, target_output_dim)
    assert type(encoded_data) is np.ndarray


def test_save_and_load(get_encoder, test_data):
    encoder = get_encoder
    assert encoder is not None

    encoded_data_control = encoder.encode(test_data)
    encoder.touch()
    encoder.save()
    assert os.path.exists(encoder.save_abspath)

    encoder_loaded = BaseExecutor.load(encoder.save_abspath)
    encoded_data_test = encoder_loaded.encode(test_data)
    np.testing.assert_array_equal(encoded_data_test, encoded_data_control)
    rm_files([encoder.save_abspath])


def test_save_and_load_config(get_encoder):
    encoder = get_encoder
    encoder.save_config()
    assert os.path.exists(encoder.config_abspath)
    encoder_loaded = BaseExecutor.load_config(encoder.config_abspath)
    assert encoder_loaded.model_path == encoder.model_path
