__copyright__ = "Copyright (c) 2021 Jina AI Limited. All rights reserved."
__license__ = "Apache-2.0"

import os

import pytest
import numpy as np

from jina.executors.metas import get_default_metas
from jina.executors import BaseExecutor

from .. import ImageKerasEncoder


@pytest.fixture
def test_metas(tmp_path):
    metas = get_default_metas()
    metas['workspace'] = str(tmp_path)
    yield metas


@pytest.fixture
def encoder(test_metas):
    yield ImageKerasEncoder(channel_axis=1, pool_strategy='max', metas=test_metas)


def test_encoding_results(encoder):
    input_dim = 96
    output_dim = 1280
    test_data = np.random.rand(2, 3, input_dim, input_dim)
    encoded_data = encoder.encode(test_data)
    assert encoded_data.shape == (2, output_dim)


def test_save_and_load(encoder):
    input_dim = 96
    test_data = np.random.rand(2, 3, input_dim, input_dim)
    encoded_data_control = encoder.encode(test_data)
    encoder.touch()
    encoder.save()
    assert os.path.exists(encoder.save_abspath)
    encoder_loaded = BaseExecutor.load(encoder.save_abspath)
    encoded_data_test = encoder_loaded.encode(test_data)
    assert encoder_loaded.channel_axis == encoder.channel_axis
    assert encoder_loaded.pool_strategy == encoder.pool_strategy
    np.testing.assert_array_equal(encoded_data_control, encoded_data_test)


def test_save_and_load_config(encoder):
    encoder.save_config()
    assert os.path.exists(encoder.config_abspath)
    encoder_loaded = BaseExecutor.load_config(encoder.config_abspath)
    assert encoder_loaded.channel_axis == encoder.channel_axis
    assert encoder_loaded.pool_strategy == encoder.pool_strategy


def test_model_name():
    encoder = ImageKerasEncoder(model_name='MobileNet')
    assert encoder.model_name == 'MobileNet'
