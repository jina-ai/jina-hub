import os
import shutil
import numpy as np
import pytest
from .. import TransformerTFEncoder
from jina.executors import BaseExecutor
from jina.executors.metas import get_default_metas


def rm_files(tmp_files):
    for k in tmp_files:
        if os.path.exists(k):
            if os.path.isfile(k):
                os.remove(k)
            elif os.path.isdir(k):
                shutil.rmtree(k, ignore_errors=False, onerror=None)


os.environ['TEST_WORKDIR'] = os.getcwd()

metas = get_default_metas()
if 'JINA_TEST_GPU' in os.environ:
    metas['on_gpu'] = True

encoders = [
    TransformerTFEncoder(
        pretrained_model_name_or_path='bert-base-uncased',
        metas=metas),
    TransformerTFEncoder(
        pooling_strategy='mean',
        pretrained_model_name_or_path='bert-base-uncased',
        metas=metas),
    TransformerTFEncoder(
        pooling_strategy='min',
        pretrained_model_name_or_path='bert-base-uncased',
        metas=metas),
    TransformerTFEncoder(
        pooling_strategy='max',
        pretrained_model_name_or_path='bert-base-uncased',
        metas=metas),
    TransformerTFEncoder(
        pretrained_model_name_or_path='xlnet-base-cased',
        metas=metas),
    TransformerTFEncoder(
        pooling_strategy='mean',
        pretrained_model_name_or_path='xlnet-base-cased',
        metas=metas),
    TransformerTFEncoder(
        pooling_strategy='min',
        pretrained_model_name_or_path='xlnet-base-cased',
        metas=metas),
    TransformerTFEncoder(
        pooling_strategy='max',
        pretrained_model_name_or_path='xlnet-base-cased',
        metas=metas),
]


@pytest.mark.parametrize('encoder', encoders)
def test_encoding_results(encoder):
    target_output_dim = 768
    test_data = np.array(['it is a good day!', 'the dog sits on the floor.'])
    encoded_data = encoder.encode(test_data)
    assert encoded_data.shape == (2, target_output_dim)
    assert not np.allclose(encoded_data[0], encoded_data[1])


@pytest.mark.parametrize('encoder', encoders)
def test_save_and_load(encoder):
    encoder.save_config()
    assert os.path.exists(encoder.config_abspath)
    test_data = np.array(['a', 'b', 'c', 'x', '!'])
    encoded_data_control = encoder.encode(test_data)

    encoder.touch()
    encoder.save()
    assert os.path.exists(encoder.save_abspath)
    encoder_loaded = BaseExecutor.load(encoder.save_abspath)
    encoded_data_test = encoder_loaded.encode(test_data)
    assert encoder_loaded.max_length == encoder.max_length
    np.testing.assert_array_equal(encoded_data_control, encoded_data_test)
    rm_files([encoder.config_abspath, encoder.save_abspath, encoder_loaded.config_abspath, encoder_loaded.save_abspath])


@pytest.mark.parametrize('encoder', encoders)
def test_save_and_load_config(encoder):
    encoder.save_config()
    assert os.path.exists(encoder.config_abspath)
    encoder_loaded = BaseExecutor.load_config(encoder.config_abspath)
    assert encoder_loaded.max_length == encoder.max_length
    rm_files([encoder_loaded.config_abspath, encoder_loaded.save_abspath])
