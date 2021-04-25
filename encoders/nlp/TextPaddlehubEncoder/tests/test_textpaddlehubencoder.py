import numpy as np
import os
import mock
import shutil
import pytest

from jina.executors.metas import get_default_metas
from .. import TextPaddlehubEncoder
from jina.executors import BaseExecutor

target_output_dim = 1024
test_data = np.array(['it is a good day!', 'the dog sits on the floor.'])
tmp_files = []


@pytest.fixture
def metas(tmpdir):
    metas = get_default_metas()
    if 'JINA_TEST_GPU' in os.environ:
        metas['on_gpu'] = True
    metas['workspace'] = str(tmpdir)
    return metas


class MockModule:
    def get_embedding(self, texts, *args, **kwargs):
        print('i am a mocker embedding')
        return [[np.random.random(target_output_dim), None]] * len(texts)


def _test_textpaddlehubencoder_encode(metas):
    encoder = TextPaddlehubEncoder(metas=metas)
    encoded_data = encoder.encode(test_data)
    assert encoded_data.shape == (2, target_output_dim)


@mock.patch('paddlehub.Module', return_value=MockModule())
def test_textpaddlehubencoder_encode(mocker, metas):
    _test_textpaddlehubencoder_encode(metas)


@mock.patch('paddlehub.Module', return_value=MockModule())
def test_textpaddlehubencoder_save_and_load(mocker, metas):
    encoder = TextPaddlehubEncoder(metas=metas)
    encoder.touch()
    encoder.save()
    assert os.path.exists(encoder.save_abspath)
    encoder_loaded = BaseExecutor.load(encoder.save_abspath)
    assert encoder_loaded.model_name == encoder.model_name


@mock.patch('paddlehub.Module', return_value=MockModule())
def test_textpaddlehubencoder_save_and_load_config(mocker, metas):
    encoder = TextPaddlehubEncoder(metas=metas)
    encoder.save_config()
    assert os.path.exists(encoder.config_abspath)
    encoder_loaded = BaseExecutor.load_config(encoder.config_abspath)
    assert encoder_loaded.model_name == encoder.model_name


def test_textpaddlehubencoder_encode_with_pretrained_model(metas):
    _test_textpaddlehubencoder_encode(metas)
