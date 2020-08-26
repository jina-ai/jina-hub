import numpy as np
import os
import mock
import shutil
import pytest

from .. import VideoPaddleEncoder
from jina.executors import BaseExecutor


target_output_dim = 2048
test_data = np.random.rand(2, 3, 3, 224, 224)
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


class MockModule:

    def context(self, *args, **kwargs):
        print('i am a mocker embedding')
        output_dim = 2048
        tmp = MockModule()
        tmp.name="videopaddle"
        inputs = [tmp]
        outputs= "output"
        from paddle.fluid.framework import Program
        model = Program()
        return inputs, outputs, model

class MockVideoPaddleEncoder:

    def to_device(self):
        import paddle.fluid as fluid
        device = fluid.CPUPlace()
        exe = fluid.Executor(device)
        return exe


def _test_videopaddlehubencoder_encode():
    encoder = VideoPaddleEncoder()
    encoded_data = encoder.encode(test_data)
    assert encoded_data.shape == (2, target_output_dim)
    add_tmpfile(encoder.save_abspath, encoder.config_abspath)
    teardown()

@mock.patch('paddlehub.Module', return_value=MockModule())
@mock.patch('..VideoPaddleEncoder', return_value=MockVideoPaddleEncoder())
def test_videopaddlehubencoder_encode(mocker):
    _test_videopaddlehubencoder_encode()

@mock.patch('paddlehub.Module', return_value=MockModule())
def test_videopaddlehubencoder_save_and_load(mocker):
    encoder = VideoPaddleEncoder()
    encoder.touch()
    encoder.save()
    assert os.path.exists(encoder.save_abspath)
    encoder_loaded = BaseExecutor.load(encoder.save_abspath)
    assert encoder_loaded.model_name == encoder.model_name
    add_tmpfile(encoder.save_abspath, encoder.config_abspath)
    teardown()


@mock.patch('paddlehub.Module', return_value=MockModule())
def test_videopaddlehubencoder_save_and_load_config(mocker):
    encoder = VideoPaddleEncoder()
    encoder.save_config()
    assert os.path.exists(encoder.config_abspath)
    encoder_loaded = BaseExecutor.load_config(encoder.config_abspath)
    assert encoder_loaded.model_name == encoder.model_name
    add_tmpfile(encoder.save_abspath, encoder.config_abspath)
    teardown()

@pytest.mark.skipif('JINA_TEST_PRETRAINED' not in os.environ, reason='skip the pretrained test if not set')
def test_videopaddlehubencoder_encode_with_pretrained_model():
    _test_videopaddlehubencoder_encode()