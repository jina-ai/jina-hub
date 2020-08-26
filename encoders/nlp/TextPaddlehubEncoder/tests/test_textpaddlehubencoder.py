import numpy as np
import os
import mock
import shutil
import pytest

from .. import TextPaddlehubEncoder
from jina.executors import BaseExecutor


target_output_dim = 1024
test_data = np.array(['it is a good day!', 'the dog sits on the floor.'])
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
    def get_embedding(self, texts, *args, **kwargs):
        print('i am a mocker embedding')
        return [[np.random.random(target_output_dim), None]] * len(texts)


def _test_textpaddlehubencoder_encode():
    encoder = TextPaddlehubEncoder()
    encoded_data = encoder.encode(test_data)
    assert encoded_data.shape == (2, target_output_dim)
    add_tmpfile(encoder.save_abspath, encoder.config_abspath)
    teardown()


@mock.patch('paddlehub.Module', return_value=MockModule())
def test_textpaddlehubencoder_encode(mocker):
    _test_textpaddlehubencoder_encode()


@mock.patch('paddlehub.Module', return_value=MockModule())
def test_textpaddlehubencoder_save_and_load(mocker):
    encoder = TextPaddlehubEncoder()
    encoder.touch()
    encoder.save()
    assert os.path.exists(encoder.save_abspath)
    encoder_loaded = BaseExecutor.load(encoder.save_abspath)
    assert encoder_loaded.model_name == encoder.model_name
    add_tmpfile(encoder.save_abspath, encoder.config_abspath)
    teardown()


@mock.patch('paddlehub.Module', return_value=MockModule())
def test_textpaddlehubencoder_save_and_load_config(mocker):
    encoder = TextPaddlehubEncoder()
    encoder.save_config()
    assert os.path.exists(encoder.config_abspath)
    encoder_loaded = BaseExecutor.load_config(encoder.config_abspath)
    assert encoder_loaded.model_name == encoder.model_name
    add_tmpfile(encoder.save_abspath, encoder.config_abspath)
    teardown()

@pytest.mark.skipif('JINA_TEST_PRETRAINED' not in os.environ, reason='skip the pretrained test if not set')
def test_textpaddlehubencoder_encode_with_pretrained_model():
    _test_textpaddlehubencoder_encode()
