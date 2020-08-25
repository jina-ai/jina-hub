import os
import pytest
import numpy as np

from .. import FlairTextEncoder
from jina.executors import BaseExecutor
from jina.executors.metas import get_default_metas


# @pytest.mark.skipif('JINA_TEST_PRETRAINED' not in os.environ, reason='skip the pretrained test if not set')
def test_encoding_results():
    metas=
    encoder = FlairTextEncoder(embeddings=('word:glove',), pooling_strategy='mean', metas=metas)
    if encoder is None:
        return
    test_data = np.array(['it is a good day!', 'the dog sits on the floor.'])
    encoded_data = encoder.encode(test_data)
    self.assertEqual(encoded_data.shape, (2, self.target_output_dim))


# @pytest.mark.skipif('JINA_TEST_PRETRAINED' not in os.environ, reason='skip the pretrained test if not set')
def test_save_and_load():
    encoder = self.get_encoder()
    if encoder is None:
        return
    test_data = np.array(['it is a good day!', 'the dog sits on the floor.'])
    encoded_data_control = encoder.encode(test_data)
    encoder.touch()
    encoder.save()
    self.assertTrue(os.path.exists(encoder.save_abspath))
    encoder_loaded = BaseExecutor.load(encoder.save_abspath)
    encoded_data_test = encoder_loaded.encode(test_data)
    self.assertEqual(encoder_loaded.max_length, encoder.max_length)
    np.testing.assert_array_equal(encoded_data_control, encoded_data_test)


# @pytest.mark.skipif('JINA_TEST_PRETRAINED' not in os.environ, reason='skip the pretrained test if not set')
def test_save_and_load_config():
    encoder = self.get_encoder()
    if encoder is None:
        return
    encoder.save_config()
    self.assertTrue(os.path.exists(encoder.config_abspath))
    encoder_loaded = BaseExecutor.load_config(encoder.config_abspath)
    self.assertEqual(encoder_loaded.max_length, encoder.max_length)