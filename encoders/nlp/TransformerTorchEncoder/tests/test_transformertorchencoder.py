import os

import numpy as np
import pytest
from jina.executors import BaseExecutor
from jina.executors.metas import get_default_metas

from .. import TransformerTorchEncoder


@pytest.fixture(scope='function')
def test_metas(tmp_path):
    metas = get_default_metas()
    metas['workspace'] = str(tmp_path)
    if 'JINA_TEST_GPU' in os.environ:
        metas['on_gpu'] = True
    yield metas


_models = [
    'sentence-transformers/distilbert-base-nli-stsb-mean-tokens',
    'sentence-transformers/bert-base-nli-stsb-mean-tokens',
    'deepset/roberta-base-squad2',
    # 'xlm-roberta-base',
    # 'xlnet-base-cased',
]


def _assert_params_equal(params_dict: dict, encoder: TransformerTorchEncoder):
    for key, val in params_dict.items():
        assert val == getattr(encoder, key)


@pytest.mark.parametrize('model_name', _models)
@pytest.mark.parametrize('pooling_strategy', ['cls', 'mean', 'max'])
@pytest.mark.parametrize('layer_index', [-1, -2, 0])
def test_encoding_results(test_metas, model_name, pooling_strategy, layer_index):
    params = {
        'pretrained_model_name_or_path': model_name,
        'pooling_strategy': pooling_strategy,
        'layer_index': layer_index
    }
    encoder = TransformerTorchEncoder(metas=test_metas, **params)

    test_data = np.array(['it is a good day!', 'the dog sits on the floor.'])
    encoded_data = encoder.encode(test_data)

    hidden_dim_sizes = {
        'sentence-transformers/distilbert-base-nli-stsb-mean-tokens': 768,
        'sentence-transformers/bert-base-nli-stsb-mean-tokens': 768,
        'deepset/roberta-base-squad2': 768,
        'xlm-roberta-base': 768,
        'xlnet-base-cased': 768,
    }
    hidden_dim_size = hidden_dim_sizes[encoder.pretrained_model_name_or_path]
    assert encoded_data.shape == (2, hidden_dim_size)

    if encoder.pooling_strategy != 'cls' or encoder.layer_index != 0:
        assert not np.allclose(encoded_data[0], encoded_data[1], rtol=1)
    else:
        assert np.allclose(encoded_data[0], encoded_data[1], atol=1e-5, rtol=1e-4)


@pytest.mark.parametrize('acceleration', ['amp', 'quant'])
def test_encoding_results_acceleration(test_metas, acceleration):
    if 'JINA_TEST_GPU' in os.environ and acceleration == 'quant':
        pytest.skip("Can't test quantization on GPU.")

    encoder = TransformerTorchEncoder(metas=test_metas, **{"acceleration": acceleration})

    test_data = np.array(['it is a good day!', 'the dog sits on the floor.'])
    encoded_data = encoder.encode(test_data)

    assert encoded_data.shape == (2, 768)
    assert not np.allclose(encoded_data[0], encoded_data[1], rtol=1)


@pytest.mark.parametrize('model_name', ['bert-base-uncased'])
@pytest.mark.parametrize('pooling_strategy', ['cls', 'mean', 'max'])
@pytest.mark.parametrize('layer_index', [-1, -2])
def test_embedding_consistency(test_metas, model_name, pooling_strategy, layer_index):
    params = {
        'pretrained_model_name_or_path': model_name,
        'pooling_strategy': pooling_strategy,
        'layer_index': layer_index,
    }
    test_data = np.array(['it is a good day!', 'the dog sits on the floor.'])

    encoder = TransformerTorchEncoder(metas=test_metas, **params)
    encoded_data = encoder.encode(test_data)

    encoded_data_file = f'tests/{model_name}-{pooling_strategy}-{layer_index}.npy'
    enc_data_loaded = np.load(encoded_data_file)

    np.testing.assert_allclose(encoded_data, enc_data_loaded, atol=1e-5, rtol=1e-6)


@pytest.mark.parametrize('model_name', _models)
@pytest.mark.parametrize('pooling_strategy', ['cls', 'mean', 'max'])
@pytest.mark.parametrize('layer_index', [-1])
def test_max_length_truncation(test_metas, model_name, pooling_strategy, layer_index):
    params = {
        'pretrained_model_name_or_path': model_name,
        'pooling_strategy': pooling_strategy,
        'layer_index': layer_index,
        'max_length': 3
    }
    encoder = TransformerTorchEncoder(metas=test_metas, **params)
    test_data = np.array(['it is a very good day!', 'it is a very sunny day!'])
    encoded_data = encoder.encode(test_data)

    np.testing.assert_allclose(encoded_data[0], encoded_data[1], atol=1e-5, rtol=1e-4)


@pytest.mark.parametrize('model_name', _models)
@pytest.mark.parametrize('pooling_strategy', ['cls', 'mean', 'max'])
@pytest.mark.parametrize('layer_index', [-1, -2])
def test_shape_single_document(test_metas, model_name, pooling_strategy, layer_index):
    params = {
        'pretrained_model_name_or_path': model_name,
        'pooling_strategy': pooling_strategy,
        'layer_index': layer_index,
        'max_length': 3
    }
    encoder = TransformerTorchEncoder(metas=test_metas, **params)
    test_data = np.array(['it is a very good day!'])
    encoded_data = encoder.encode(test_data)
    assert len(encoded_data.shape) == 2
    assert encoded_data.shape[0] == 1


@pytest.mark.parametrize('model_name', _models)
@pytest.mark.parametrize('pooling_strategy', ['min'])
@pytest.mark.parametrize('layer_index', [-2])
def test_save_and_load(test_metas, model_name, pooling_strategy, layer_index):
    params = {
        'pretrained_model_name_or_path': model_name,
        'pooling_strategy': pooling_strategy,
        'layer_index': layer_index
    }
    encoder = TransformerTorchEncoder(metas=test_metas, **params)
    test_data = np.array(['a', 'b', 'c', 'x', '!'])
    encoded_data_control = encoder.encode(test_data)

    encoder.touch()
    encoder.save()
    assert os.path.exists(encoder.save_abspath)

    encoder_loaded = BaseExecutor.load(encoder.save_abspath)
    _assert_params_equal(params, encoder_loaded)

    encoded_data_test = encoder_loaded.encode(test_data)
    np.testing.assert_array_equal(encoded_data_control, encoded_data_test)


@pytest.mark.parametrize('model_name', _models)
@pytest.mark.parametrize('pooling_strategy', ['min'])
@pytest.mark.parametrize('layer_index', [-2])
def test_save_and_load_config(test_metas, model_name, pooling_strategy, layer_index):
    params = {
        'pretrained_model_name_or_path': model_name,
        'pooling_strategy': pooling_strategy,
        'layer_index': layer_index
    }
    encoder = TransformerTorchEncoder(metas=test_metas, **params)

    encoder.save_config()
    _assert_params_equal(params, encoder)
    assert os.path.exists(encoder.config_abspath)

    encoder_loaded = BaseExecutor.load_config(encoder.config_abspath)
    _assert_params_equal(params, encoder_loaded)


@pytest.mark.parametrize('layer_index', [-100, 100])
def test_wrong_layer_index(test_metas, layer_index):
    params = {'layer_index': layer_index}
    encoder = TransformerTorchEncoder(metas=test_metas, **params)

    encoder.layer_index = layer_index
    test_data = np.array(['it is a good day!', 'the dog sits on the floor.'])
    with pytest.raises(ValueError):
        _ = encoder.encode(test_data)


def test_wrong_pooling_strategy():
    with pytest.raises(NotImplementedError):
        TransformerTorchEncoder(pooling_strategy='wrong')


def test_wrong_pooling_acceleration():
    with pytest.raises(NotImplementedError):
        TransformerTorchEncoder(acceleration='wrong')


@pytest.mark.parametrize(
    'params',
    [{'pooling_strategy': 'cls', 'pretrained_model_name_or_path': 'gpt2'}],
)
def test_no_cls_token(test_metas, params):
    encoder = TransformerTorchEncoder(metas=test_metas, **params)
    test_data = np.array(['it is a good day!', 'the dog sits on the floor.'])
    with pytest.raises(ValueError):
        _ = encoder.encode(test_data)
