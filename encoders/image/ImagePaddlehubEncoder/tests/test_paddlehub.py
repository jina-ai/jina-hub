import os
import mock
import shutil
import numpy as np
import pytest
from .. import ImagePaddlehubEncoder
from jina.executors.metas import get_default_metas
from jina.executors import BaseExecutor

input_dim = 224
target_output_dim = 2048
num_doc = 2
test_data = np.random.rand(num_doc, 3, input_dim, input_dim)
tmp_files = []


def teardown():
    for k in tmp_files:
        if os.path.exists(k):
            if os.path.isfile(k):
                os.remove(k)
            elif os.path.isdir(k):
                shutil.rmtree(k, ignore_errors=False, onerror=None)


def add_tmpfile(*path):
    tmp_files.extend(path)


def get_encoder():
    metas = get_default_metas()
    if 'JINA_TEST_GPU' in os.environ:
        metas['on_gpu'] = True
    return ImagePaddlehubEncoder(metas=metas)


class MockModule:
    def get_embedding(self, texts, *args, **kwargs):
        return [[np.random.random(target_output_dim), None]] * len(texts)

    def context(self, *args, **kwargs):
        return {'image': MockTensor('input')}, {'feature_map': MockTensor('output')}, MockModel()


class MockTensor:
    def __init__(self, name):
        self.name = name


class MockModel:
    pass


class MockExecutor:
    def __init__(self, *args, **kwargs):
        pass

    def close(self):
        pass

    def run(self, feed, *args, **kwargs):
        feature_map = np.random.random((num_doc, target_output_dim))
        return feature_map, None


class MockCUDADevice:
    pass


class MockCPUDevice:
    pass


def _test_imagepaddlehubencoder_encode(*args, **kwargs):
    encoder = get_encoder()
    encoded_data = encoder.encode(test_data)
    assert encoded_data.shape == (num_doc, target_output_dim)
    add_tmpfile(encoder.save_abspath, encoder.config_abspath)
    teardown()


@mock.patch('paddlehub.Module', return_value=MockModule())
@mock.patch('paddle.fluid.Executor', return_value=MockExecutor())
@mock.patch('paddle.fluid.CUDAPlace', return_value=MockCUDADevice())
@mock.patch('paddle.fluid.CPUPlace', return_value=MockCPUDevice())
def test_imagepaddlehubencoder_encode(*args, **kwargs):
    _test_imagepaddlehubencoder_encode(*args, **kwargs)


@mock.patch('paddlehub.Module', return_value=MockModule())
@mock.patch('paddle.fluid.Executor', return_value=MockExecutor())
@mock.patch('paddle.fluid.CUDAPlace', return_value=MockCUDADevice())
@mock.patch('paddle.fluid.CPUPlace', return_value=MockCPUDevice())
def test_imagepaddlehubencoder_save_and_load(*args, **kwargs):
    encoder = get_encoder()
    encoder.touch()
    encoder.save()
    assert os.path.exists(encoder.save_abspath)
    encoder_loaded = BaseExecutor.load(encoder.save_abspath)
    assert encoder_loaded.model_name == encoder.model_name
    add_tmpfile(encoder.save_abspath, encoder.config_abspath)
    teardown()


@mock.patch('paddlehub.Module', return_value=MockModule())
@mock.patch('paddle.fluid.Executor', return_value=MockExecutor())
@mock.patch('paddle.fluid.CUDAPlace', return_value=MockCUDADevice())
@mock.patch('paddle.fluid.CPUPlace', return_value=MockCPUDevice())
def test_imagepaddlehubencoder_save_and_load_config(*args, **kwargs):
    encoder = get_encoder()
    encoder.save_config()
    assert os.path.exists(encoder.config_abspath)
    encoder_loaded = BaseExecutor.load_config(encoder.config_abspath)
    assert encoder_loaded.model_name == encoder.model_name
    add_tmpfile(encoder.save_abspath, encoder.config_abspath)
    teardown()


@pytest.mark.skipif('JINA_TEST_PRETRAINED' not in os.environ, reason='skip the pretrained test if not set')
def test_imagepaddlehubencoder_encode_with_pretrained_model():
    _test_imagepaddlehubencoder_encode()
