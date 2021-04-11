import os
import pytest
import pickle
import shutil

import numpy as np

from .. import TruncatedSVDEncoder
from jina.executors import BaseExecutor
from jina.executors.encoders.numeric import TransformEncoder

tmp_model_filename = "truncated_svd.model"


def get_encoder(train_data, target_output_dim):
    from sklearn.decomposition import TruncatedSVD

    model = TruncatedSVD(n_components=target_output_dim)
    pickle.dump(model.fit(train_data), open(tmp_model_filename, "wb"))
    return TransformEncoder(model_path=tmp_model_filename)


@pytest.fixture(scope="function")
def train_data():
    """
    Create an array of the given shape and populate it with random
    samples from a uniform distribution over [0, 1).

    :return: a `B x T` numpy ``ndarray``, `B` is the size of the batch
    """
    batch_size = 2000
    input_dim = 28
    train_data = np.random.rand(batch_size, input_dim)

    return train_data


@pytest.fixture(scope="function")
def test_data():
    """
    Create an array of the given shape and populate it with random
    samples from a uniform distribution over [0, 1).

    :return: a `B x T` numpy ``ndarray``, `B` is the size of the batch
    """
    batch_size = 10
    input_dim = 28
    test_data = np.random.rand(batch_size, input_dim)

    return test_data


@pytest.mark.parametrize("target_output_dim", [2])
def test_truncated_svd_encoder_train(train_data, test_data, target_output_dim):
    encoder = TruncatedSVDEncoder(output_dim=target_output_dim)
    encoder.train(train_data)
    encoded_data = encoder.encode(test_data)

    assert encoded_data.shape == (test_data.shape[0], target_output_dim)
    assert type(encoded_data) == np.ndarray


@pytest.mark.parametrize("target_output_dim", [2])
def test_truncated_svd_save_and_load(train_data, test_data, target_output_dim):
    encoder = get_encoder(train_data, target_output_dim)
    encoded_data_orig = encoder.encode(test_data)

    encoder.touch()
    encoder.save()
    assert os.path.exists(encoder.save_abspath)

    encoder_loaded = BaseExecutor.load(encoder.save_abspath)
    encoded_data_test = encoder_loaded.encode(test_data)
    np.testing.assert_array_equal(encoded_data_test, encoded_data_orig)

    shutil.rmtree(f"{encoder.workspace_name}-0")
    os.remove(tmp_model_filename)
