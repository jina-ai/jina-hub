import os
import numpy as np
import shutil
import mock

from .. import FlairTextEncoder
from jina.executors import BaseExecutor
from jina.executors.metas import get_default_metas

target_output_dim = 100


class MockModule:
    def __init__(self):
        print('i am a mocker module')

    def get_embedding(self, texts, *args, **kwargs):
        print('i am a mocker embedding')
        return [[np.random.random(target_output_dim), None]] * len(texts)


def get_metas():
    metas = get_default_metas()
    if 'JINA_TEST_GPU' in os.environ:
        metas['on_gpu'] = True
    return metas

def rm_files(tmp_files):
    for file in tmp_files:
        if file and os.path.exists(file):
            if os.path.isfile(file):
                os.remove(file)
            elif os.path.isdir(file):
                shutil.rmtree(file, ignore_errors=False, onerror=None)

@mock.patch('paddlehub.Module', return_value=MockModule())
def test_encoding_results():
    metas = get_metas()
    encoder = FlairTextEncoder(embeddings=('word:glove',), pooling_strategy='mean', metas=metas)
    test_data = np.array(['it is a good day!', 'the dog sits on the floor.'])
    encoded_data = encoder.encode(test_data)
    assert encoded_data.shape == (2, target_output_dim)
    rm_files([encoder.config_abspath, encoder.save_abspath])


@mock.patch('paddlehub.Module', return_value=MockModule())
def test_save_and_load():
    metas = get_metas()
    encoder = FlairTextEncoder(embeddings=('word:glove',), pooling_strategy='mean', metas=metas)
    test_data = np.array(['it is a good day!', 'the dog sits on the floor.'])
    encoded_data_control = encoder.encode(test_data)
    encoder.touch()
    encoder.save()
    assert os.path.exists(encoder.save_abspath)
    encoder_loaded = BaseExecutor.load(encoder.save_abspath)
    encoded_data_test = encoder_loaded.encode(test_data)
    assert encoder_loaded.embeddings == encoder.embeddings
    np.testing.assert_array_equal(encoded_data_control, encoded_data_test)
    rm_files([encoder.config_abspath, encoder.save_abspath])


@mock.patch('paddlehub.Module', return_value=MockModule())
def test_save_and_load_config():
    metas = get_metas()
    encoder = FlairTextEncoder(embeddings=('word:glove',), pooling_strategy='mean', metas=metas)
    encoder.save_config()
    assert os.path.exists(encoder.config_abspath)
    encoder_loaded = BaseExecutor.load_config(encoder.config_abspath)
    assert encoder_loaded.max_length == encoder.max_length
    rm_files([encoder.config_abspath, encoder.save_abspath])





