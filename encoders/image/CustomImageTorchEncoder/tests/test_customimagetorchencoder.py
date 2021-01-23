import pytest
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from jina.executors.metas import get_default_metas
from jina.excepts import PretrainedModelFileDoesNotExist
from jina.executors import BaseExecutor
from .. import CustomImageTorchEncoder


class ExampleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 10, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(10, 16, 5)
        self.fc1 = nn.Linear(16 * 53 * 53, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 53 * 53)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


@pytest.fixture(scope='function', autouse=True)
def test_metas(tmpdir):
    metas = get_default_metas()
    if 'JINA_TEST_GPU' in os.environ:
        metas['on_gpu'] = True
    metas['workspace'] = str(tmpdir)
    yield metas


@pytest.fixture
def encoder(test_metas):
    path = os.path.join(test_metas['workspace'], 'model.pth')
    model = ExampleNet()
    torch.save(model, path)
    return CustomImageTorchEncoder(model_path=path, layer_name='conv1', pool_strategy='max', metas=test_metas)


@pytest.fixture
def default_encoder(test_metas):
    return CustomImageTorchEncoder(metas=test_metas)


def test_encoding_results(encoder):
    output_dim = 10
    input_dim = 224
    test_data = np.random.rand(2, 3, input_dim, input_dim)
    encoded_data = encoder.encode(test_data)
    assert encoded_data.shape == (2, output_dim)


def test_encoding_results_default(default_encoder):
    output_dim = 1280
    input_dim = 224
    test_data = np.random.rand(2, 3, input_dim, input_dim)
    encoded_data = default_encoder.encode(test_data)
    assert encoded_data.shape == (2, output_dim)


def test_save_and_load(encoder):
    input_dim = 224
    test_data = np.random.rand(2, 3, input_dim, input_dim)
    encoded_data_control = encoder.encode(test_data)
    encoder.touch()
    encoder.save()
    assert os.path.exists(encoder.save_abspath)
    encoder_loaded = BaseExecutor.load(encoder.save_abspath)
    encoded_data_test = encoder_loaded.encode(test_data)
    assert encoder_loaded.channel_axis == encoder.channel_axis
    assert encoder_loaded.pool_strategy == encoder.pool_strategy
    assert encoder_loaded.layer_name == encoder.layer_name
    np.testing.assert_array_equal(encoded_data_control, encoded_data_test)


def test_save_and_load_config(encoder):
    encoder.save_config()
    assert os.path.exists(encoder.config_abspath)
    encoder_loaded = BaseExecutor.load_config(encoder.config_abspath)
    assert encoder_loaded.channel_axis == encoder.channel_axis
    assert encoder_loaded.pool_strategy == encoder.pool_strategy
    assert encoder_loaded.layer_name == encoder.layer_name


def test_raise_exception():
    with pytest.raises(PretrainedModelFileDoesNotExist):
        assert CustomImageTorchEncoder(model_path=None)
