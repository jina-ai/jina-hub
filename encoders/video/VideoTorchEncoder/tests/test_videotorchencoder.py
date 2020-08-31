import os
import shutil
import mock
import numpy as np
import pytest

from .. import VideoTorchEncoder
from jina.executors.metas import get_default_metas
from jina.executors import BaseExecutor
from typing import TypeVar

batch_size = 2
input_dim = 224
output_dim = 512
channel = 3
num_frames = 10


def rm_files(file_paths):
    for file_path in file_paths:
        if os.path.exists(file_path):
            if os.path.isfile(file_path):
                os.remove(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path, ignore_errors=False, onerror=None)


class MockModels:
    def r3d_18(pretrained=True, *args, **kwargs):
        return MockModel()


class MockModel:
    def eval(self):
        return self

    def to(self, *args, **kwargs):
        pass

class MockFeature:
    def numpy(self):
        np.random.seed(0)
        return np.random.random((batch_size, output_dim))

    def detach(self):
        return self

    def cpu(self):
        return self


def get_encoder(*args, **kwargs):
    metas = get_default_metas()
    if 'JINA_TEST_GPU' in os.environ:
        metas['on_gpu'] = True
    return VideoTorchEncoder(metas=metas)

def _test_encoding_results(*args, **kwargs):
    encoder = get_encoder(*args, **kwargs)
    test_data = np.random.rand(batch_size, num_frames, channel, input_dim, input_dim)
    encoded_data = encoder.encode(test_data)
    assert encoded_data.shape == (batch_size, output_dim)
    rm_files([encoder.save_abspath, encoder.config_abspath])

@mock.patch('torchvision.models.video', return_value=MockModels())
@mock.patch.object(VideoTorchEncoder, '_get_features', return_value=MockFeature())
def test_encoding_results(*args, **kwargs):
    _test_encoding_results(*args, **kwargs)

@mock.patch('torchvision.models.video', return_value=MockModels())
@mock.patch.object(VideoTorchEncoder, '_get_features', return_value=MockFeature())
def test_save_and_load(*args, **kwargs):
    encoder = get_encoder()
    test_data = np.random.rand(batch_size, num_frames, channel, input_dim, input_dim)
    encoded_data_control = encoder.encode(test_data)
    encoder.touch()
    encoder.save()
    assert os.path.exists(encoder.save_abspath)
    encoder_loaded = BaseExecutor.load(encoder.save_abspath)
    encoded_data_test = encoder_loaded.encode(test_data)
    assert encoder_loaded.channel_axis == encoder.channel_axis
    np.testing.assert_array_equal(encoded_data_control, encoded_data_test)
    rm_files([encoder.save_abspath, encoder.config_abspath])

@mock.patch('torchvision.models.video', return_value=MockModels())
@mock.patch.object(VideoTorchEncoder, '_get_features', return_value=MockFeature())
def test_save_and_load_config(*args, **kwargs):
    encoder = get_encoder()
    encoder.save_config()
    assert os.path.exists(encoder.config_abspath)
    encoder_loaded = BaseExecutor.load_config(encoder.config_abspath)
    assert encoder_loaded.channel_axis == encoder.channel_axis
    rm_files([encoder.save_abspath, encoder.config_abspath])

def test_pool_fn():
    test_data = np.random.rand(batch_size, num_frames, channel, input_dim, input_dim)
    encoder = get_encoder()
    encoded_data = encoder.pool_fn(test_data, axis=(2, 3))
    assert encoded_data.ndim == test_data.ndim - 2

@pytest.mark.skipif('JINA_TEST_PRETRAINED' not in os.environ, reason='skip the pretrained test if not set')
def test_video_torch_encode_with_pretrained_model():
    _test_encoding_results()