import os
import numpy as np
import shutil
import mock
import pytest

from .. import FlairTextEncoder
from jina.executors import BaseExecutor
from jina.executors.metas import get_default_metas

target_output_dim = 100


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


test_data = np.array(['it is a good day!', 'the dog sits on the floor.'])


class MockEmbedding:
    pass


class MockDocumentEmbedding:
    def embed(self, sentences):
        return np.random.random((len(sentences), target_output_dim))


class MockSentence:
    @property
    def embedding(self):
        return np.random.random((1, target_output_dim))


def _test_encoding_results(*args, **kwargs):
    encoder = FlairTextEncoder(embeddings=('word:glove',), pooling_strategy='mean')
    encoded_data = encoder.encode(test_data)
    assert encoded_data.shape == (2, target_output_dim)
    rm_files([encoder.config_abspath, encoder.save_abspath])


@mock.patch('flair.device', return_value='')
@mock.patch('flair.embeddings.WordEmbeddings', return_value=MockEmbedding())
@mock.patch('flair.embeddings.DocumentPoolEmbeddings', return_value=MockDocumentEmbedding())
@mock.patch('flair.data.Sentence', return_value=MockSentence())
def test_encoding_results_local(*args, **kwargs):
    _test_encoding_results(*args, **kwargs)


@mock.patch('flair.device', return_value='')
@mock.patch('flair.embeddings.WordEmbeddings', return_value=MockEmbedding())
@mock.patch('flair.embeddings.DocumentPoolEmbeddings', return_value=MockDocumentEmbedding())
@mock.patch('flair.data.Sentence', return_value=MockSentence())
def test_save_and_load(*args, **kwargs):
    encoder = FlairTextEncoder(embeddings=('word:glove',), pooling_strategy='mean')
    encoder.touch()
    encoder.save()
    assert os.path.exists(encoder.save_abspath)
    encoder_loaded = BaseExecutor.load(encoder.save_abspath)
    assert encoder_loaded.embeddings == encoder.embeddings
    rm_files([encoder.config_abspath, encoder.save_abspath])


@mock.patch('flair.device', return_value='')
@mock.patch('flair.embeddings.WordEmbeddings', return_value=MockEmbedding())
@mock.patch('flair.embeddings.FlairEmbeddings', return_value=MockEmbedding())
@mock.patch('flair.embeddings.DocumentPoolEmbeddings', return_value=MockDocumentEmbedding())
@mock.patch('flair.data.Sentence', return_value=MockSentence())
def test_save_and_load_config(*args, **kwargs):
    encoder = FlairTextEncoder(embeddings=('flair:news-forward',), pooling_strategy='mean')
    encoder.save_config()
    assert os.path.exists(encoder.config_abspath)
    encoder_loaded = BaseExecutor.load_config(encoder.config_abspath)
    assert encoder_loaded.max_length == encoder.max_length
    rm_files([encoder.config_abspath, encoder.save_abspath])


@pytest.mark.skipif('JINA_TEST_PRETRAINED' not in os.environ, reason='skip the pretrained test if not set')
def test_encoding_results(*args, **kwargs):
    _test_encoding_results(*args, **kwargs)
