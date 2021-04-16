import os
import pickle

import numpy as np
import pytest
from sklearn.decomposition import FastICA

from jina.executors import BaseExecutor
from jina.executors.metas import get_default_metas

from .. import FastICAEncoder


@pytest.fixture(scope="function", autouse=True)
def metas(tmpdir):
    metas = get_default_metas()
    metas['workspace'] = str(tmpdir)
    yield metas


@pytest.fixture(scope="function")
def train_data():
    """
    Create an array of the given shape and populate it with random samples from a uniform distribution over [0, 1).
    :return: a `B x T` numpy ``ndarray``, `B` is the size of the batch
    """
    batch_size = 2000
    input_dim = 28
    train_data = np.random.rand(batch_size, input_dim)

    return train_data


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


def get_encoder(metas, train_data, target_output_dim):
    tmpdir = metas['workspace']
    model_path = os.path.join(tmpdir, 'fastica_model.model')

    model = FastICA(n_components=target_output_dim,
                    whiten=True,
                    max_iter=10)
    model.fit(train_data)
    pickle.dump(model, open(model_path, 'wb'))

    return FastICAEncoder(model_path=model_path)


@pytest.mark.parametrize('target_output_dim', [2])
def test_encoding_results(metas, train_data, test_data, target_output_dim):
    expected_batch_size = test_data.shape[0]
    encoder = get_encoder(metas, train_data, target_output_dim)
    encoded_data = encoder.encode(test_data)
    assert encoded_data.shape == (expected_batch_size, target_output_dim)
    assert type(encoded_data) is np.ndarray


@pytest.mark.parametrize('target_output_dim', [2])
def test_save_and_load(metas, train_data, test_data, target_output_dim):
    encoder = get_encoder(metas, train_data, target_output_dim)
    encoded_data_control = encoder.encode(test_data)
    encoder.touch()
    encoder.save()
    assert os.path.exists(encoder.save_abspath)

    encoder_loaded = BaseExecutor.load(encoder.save_abspath)
    encoded_data_test = encoder_loaded.encode(test_data)
    np.testing.assert_array_equal(encoded_data_test, encoded_data_control)
