import os
import numpy as np
import pytest
import random
import string
from .. import TransformerTorchEncoder
from jina.executors import BaseExecutor
from jina.executors.metas import get_default_metas


@pytest.fixture(scope='function')
def random_workspace_name():
    """Generate a random workspace name with digits and letters."""
    rand = ''.join(random.choices(string.ascii_uppercase + string.digits, k=6))
    return f'JINA_TEST_WORKSPACE_{rand}'


@pytest.fixture(scope='function')
def test_metas(tmpdir, random_workspace_name):
    os.environ[random_workspace_name] = str(tmpdir)
    metas = get_default_metas()
    metas['workspace'] = os.environ[random_workspace_name]
    if 'JINA_TEST_GPU' in os.environ:
        metas['on_gpu'] = True
    yield metas
    del os.environ[random_workspace_name]


encoders_parameters = [
    {
        "pretrained_model_name_or_path": 'sentence-transformers/distilbert-base-nli-stsb-mean-tokens',
        "model_save_path": 'distilbert-base-nli-stsb-mean-tokens',
    },
    {
        "pooling_strategy": 'auto',
        "pretrained_model_name_or_path": 'distilbert-base-uncased',
        "model_save_path": 'distilbert-base-uncased-mean',
    },
    {
        "pooling_strategy": 'min',
        "pretrained_model_name_or_path": 'distilbert-base-uncased',
        "model_save_path": 'distilbert-base-uncased-min',
    },
    {
        "pooling_strategy": 'max',
        "pretrained_model_name_or_path": 'distilbert-base-uncased',
        "model_save_path": 'distilbert-base-uncased-max',
    },
    {
        "pretrained_model_name_or_path": 'xlnet-base-cased',
        "model_save_path": 'xlnet-base-cased',
    },
    {
        "pooling_strategy": 'auto',
        "pretrained_model_name_or_path": 'xlnet-base-cased',
        "model_save_path": 'xlnet-base-cased-mean',
    },
    {
        "pooling_strategy": 'min',
        "pretrained_model_name_or_path": 'xlnet-base-cased',
        "model_save_path": 'xlnet-base-cased-min',
    },
    {
        "pooling_strategy": 'max',
        "pretrained_model_name_or_path": 'xlnet-base-cased',
        "model_save_path": 'xlnet-base-cased-max',
    },
    {
        "pooling_strategy": 'mean',
        "pretrained_model_name_or_path": 'distilbert-base-uncased',
        "model_save_path": 'distilbert-base-uncased-mean',
        "layer_index": -2,
    },
    {
        "pooling_strategy": 'mean',
        "pretrained_model_name_or_path": 'distilbert-base-cased',
        "model_save_path": 'distilbert-base-cased-mean',
        "max_length": 100
    },
]


@pytest.fixture
def encoder(request, test_metas):
    return TransformerTorchEncoder(metas=test_metas, **request.param)


@pytest.mark.parametrize('encoder', encoders_parameters, indirect=['encoder'])
def test_encoding_results(encoder):
    target_output_dim = 768
    test_data = np.array(['it is a good day!', 'the dog sits on the floor.'])
    encoded_data = encoder.encode(test_data)
    assert encoded_data.shape == (2, target_output_dim)
    assert not np.allclose(encoded_data[0], encoded_data[1])


@pytest.mark.parametrize('encoder', encoders_parameters, indirect=['encoder'])
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


@pytest.mark.parametrize('encoder', encoders_parameters, indirect=['encoder'])
def test_save_and_load_config(encoder):
    encoder.save_config()
    assert os.path.exists(encoder.config_abspath)
    encoder_loaded = BaseExecutor.load_config(encoder.config_abspath)
    assert encoder_loaded.max_length == encoder.max_length


@pytest.mark.parametrize('encoder', encoders_parameters[5:6], indirect=['encoder'])
def test_parameter_override(encoder):
    encoder_preset = encoders_parameters[5]
    assert encoder.pretrained_model_name_or_path == encoder_preset['pretrained_model_name_or_path']
    assert encoder.pooling_strategy == encoder_preset['pooling_strategy']
    assert encoder.model_save_path == encoder_preset['model_save_path']

@pytest.mark.parametrize('layer_index', [-100, 100])
@pytest.mark.parametrize('encoder', encoders_parameters[0:1], indirect=['encoder'])
def test_wrong_layer_index(encoder, layer_index):
    encoder.layer_index = layer_index
    test_data = np.array(['it is a good day!', 'the dog sits on the floor.'])
    with pytest.raises(ValueError, match=f'Invalid value {encoder.layer_index}'):
        encoded_data = encoder.encode(test_data)

def test_max_length(test_metas):
    encoder = TransformerTorchEncoder(metas=test_metas, max_length=3)
    test_data = np.array(['it is a very good day!', 'it is a very sunny day!'])
    encoded_data = encoder.encode(test_data)

    np.testing.assert_allclose(encoded_data[0], encoded_data[1])
