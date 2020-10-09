import os
import numpy as np
import pytest

from cvae import cvae

from jina.executors import BaseExecutor
from jina.executors.metas import get_default_metas

from .. import CompressionVaeEncoder

target_output_dim = 2


@pytest.fixture(scope='function', autouse=True)
def metas(tmpdir):
    os.environ['TEST_WORKSPACE'] = str(tmpdir)
    metas = get_default_metas()
    metas['workspace'] = os.environ['TEST_WORKSPACE']
    yield metas
    del os.environ['TEST_WORKSPACE']


def get_encoder(metadata, test_data):
    tmpdir = metadata['workspace']
    model_path = os.path.join(tmpdir, 'model')
    data_path = os.path.join(tmpdir, 'data')
    os.mkdir(model_path)
    os.mkdir(data_path)

    for idx, features in enumerate(test_data):
        np.save(os.path.join(data_path, str(idx)), features)

    # Train the CVAE on the test data to build a model saved in `logdir`.
    model = cvae.CompressionVAE(data_path,
                                dim_latent=target_output_dim,
                                logdir=model_path)
    model.train()

    return CompressionVaeEncoder(model_path=model_path, metas=metadata)


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


def test_encoding_results(metas, test_data):
    expected_batch_size = test_data.shape[0]
    encoder = get_encoder(metas, test_data)
    encoded_data = encoder.encode(test_data)
    assert encoded_data.shape == (expected_batch_size, target_output_dim)
    assert type(encoded_data) is np.ndarray


def test_save_and_load(metas, test_data):
    encoder = get_encoder(metas, test_data)
    encoded_data_control = encoder.encode(test_data)
    encoder.touch()
    encoder.save()
    assert os.path.exists(encoder.save_abspath)

    encoder_loaded = BaseExecutor.load(encoder.save_abspath)
    encoded_data_test = encoder_loaded.encode(test_data)
    np.testing.assert_array_equal(encoded_data_test, encoded_data_control)
