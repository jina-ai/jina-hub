import tempfile
import os
import shutil

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from jina.executors.metas import get_default_metas
from .. import CustomImageTorchEncoder
from jina.executors import BaseExecutor


def rm_files(file_paths):
    for file_path in file_paths:
        if os.path.exists(file_path):
            if os.path.isfile(file_path):
                os.remove(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path, ignore_errors=False, onerror=None)


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


def get_encoder():
    metas = get_default_metas()
    if 'JINA_TEST_GPU' in os.environ:
        metas['on_gpu'] = True
    path = tempfile.NamedTemporaryFile().name
    model = ExampleNet()
    torch.save(model, path)
    return CustomImageTorchEncoder(model_path=path, layer_name='conv1', metas=metas)


def test_encoding_results():
    output_dim = 10
    input_dim = 224
    encoder = get_encoder()
    test_data = np.random.rand(2, 3, input_dim, input_dim)
    encoded_data = encoder.encode(test_data)
    assert encoded_data.shape == (2, output_dim)
    rm_files([encoder.save_abspath, encoder.config_abspath, encoder.model_path])


def test_save_and_load():
    input_dim = 224
    encoder = get_encoder()
    test_data = np.random.rand(2, 3, input_dim, input_dim)
    encoded_data_control = encoder.encode(test_data)
    encoder.touch()
    encoder.save()
    assert os.path.exists(encoder.save_abspath)
    encoder_loaded = BaseExecutor.load(encoder.save_abspath)
    encoded_data_test = encoder_loaded.encode(test_data)
    assert encoder_loaded.channel_axis == encoder.channel_axis
    np.testing.assert_array_equal(encoded_data_control, encoded_data_test)
    rm_files([encoder.save_abspath, encoder.config_abspath, encoder.model_path])
