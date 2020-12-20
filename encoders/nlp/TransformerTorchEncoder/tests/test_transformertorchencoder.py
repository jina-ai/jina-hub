import os
from itertools import product
import random
import string

import numpy as np
import pytest
from jina.executors import BaseExecutor
from jina.executors.metas import get_default_metas

from .. import TransformerTorchEncoder


@pytest.fixture(scope="function")
def test_metas(tmp_path):
    metas = get_default_metas()
    metas["workspace"] = str(tmp_path)
    if "JINA_TEST_GPU" in os.environ:
        metas["on_gpu"] = True
    yield metas


@pytest.fixture
def encoder(request, test_metas):
    request.param["model_save_path"] = (
        request.param["pretrained_model_name_or_path"].replace(
            "sentence-transformers/", ""
        )
        + f'-{request.param["pooling_strategy"]}'
    )
    return TransformerTorchEncoder(metas=test_metas, **request.param)


_params_dict = {
    "pretrained_model_name_or_path": [
        "sentence-transformers/distilbert-base-nli-stsb-mean-tokens",
        "sentence-transformers/bert-base-nli-stsb-mean-tokens",
        "sentence-transformers/xlm-r-100langs-bert-base-nli-stsb-mean-tokens",
        "sentence-transformers/roberta-base-nli-stsb-mean-tokens",
        "xlnet-base-cased",
    ],
    "pooling_strategy": ["cls", "mean", "max"],
    "layer_index": [-1, -2, 0],
}

# Create a cartesian product of the param lists
encoder_params = [
    dict(zip(_params_dict.keys(), vals)) for vals in product(*_params_dict.values())
]
encoder_params_ind_1 = filter(lambda x: x["layer_index"] == -1, encoder_params)


def _assert_params_equal(params_dict: dict, encoder: TransformerTorchEncoder):
    for key, val in params_dict.items():
        assert val == getattr(encoder, key)


@pytest.mark.parametrize("encoder", encoder_params, indirect=True)
def test_encoding_results(encoder):
    test_data = np.array(["it is a good day!", "the dog sits on the floor."])
    encoded_data = encoder.encode(test_data)

    hidden_dim_sizes = {
        "sentence-transformers/distilbert-base-nli-stsb-mean-tokens": 768,
        "sentence-transformers/bert-base-nli-stsb-mean-tokens": 768,
        "sentence-transformers/xlm-r-100langs-bert-base-nli-stsb-mean-tokens": 768,
        "sentence-transformers/roberta-base-nli-stsb-mean-tokens": 768,
        "xlnet-base-cased": 768,
    }
    hidden_dim_size = hidden_dim_sizes[encoder.pretrained_model_name_or_path]
    assert encoded_data.shape == (2, hidden_dim_size)

    if encoder.pooling_strategy != "cls" or encoder.layer_index != 0:
        assert not np.allclose(encoded_data[0], encoded_data[1], rtol=1)
    else:
        assert np.allclose(encoded_data[0], encoded_data[1], atol=1e-5, rtol=1e-4)


@pytest.mark.parametrize("encoder", encoder_params_ind_1, indirect=True)
def test_max_length_truncation(encoder):
    encoder.max_length = 3
    test_data = np.array(["it is a very good day!", "it is a very sunny day!"])
    encoded_data = encoder.encode(test_data)

    np.testing.assert_allclose(encoded_data[0], encoded_data[1], atol=1e-5, rtol=1e-4)


@pytest.mark.parametrize(
    "encoder, params", list(zip(encoder_params, encoder_params)), indirect=["encoder"]
)
def test_save_and_load(encoder, params):
    encoder.save_config()
    _assert_params_equal(params, encoder)
    assert os.path.exists(encoder.config_abspath)

    test_data = np.array(["a", "b", "c", "x", "!"])
    encoded_data_control = encoder.encode(test_data)

    encoder.touch()
    encoder.save()
    assert os.path.exists(encoder.save_abspath)

    encoder_loaded = BaseExecutor.load(encoder.save_abspath)
    _assert_params_equal(params, encoder_loaded)

    encoded_data_test = encoder_loaded.encode(test_data)
    np.testing.assert_array_equal(encoded_data_control, encoded_data_test)


@pytest.mark.parametrize(
    "encoder, params", list(zip(encoder_params, encoder_params)), indirect=["encoder"]
)
def test_save_and_load_config(encoder, params):
    encoder.save_config()
    _assert_params_equal(params, encoder)
    assert os.path.exists(encoder.config_abspath)

    encoder_loaded = BaseExecutor.load_config(encoder.config_abspath)
    _assert_params_equal(params, encoder_loaded)


@pytest.mark.parametrize("layer_index", [-100, 100])
@pytest.mark.parametrize("encoder", encoder_params[0], indirect=["encoder"])
def test_wrong_layer_index(encoder, layer_index):
    encoder.layer_index = layer_index
    test_data = np.array(["it is a good day!", "the dog sits on the floor."])
    with pytest.raises(ValueError):
        encoded_data = encoder.encode(test_data)


def test_wrong_pooling_strategy():
    with pytest.raises(NotImplementedError):
        TransformerTorchEncoder(pooling_strategy="wrong")


@pytest.mark.parametrize(
    "encoder",
    [{"pooling_strategy": "cls", "pretrained_model_name_or_path": "gpt2"}],
    indirect=True,
)
def test_no_cls_token(encoder):
    test_data = np.array(["it is a good day!", "the dog sits on the floor."])
    with pytest.raises(ValueError):
        encoder.encode(test_data)
