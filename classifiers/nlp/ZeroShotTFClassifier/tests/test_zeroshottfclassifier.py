import numpy as np
import os
import pytest

from jina.executors import BaseExecutor
from jina.executors.metas import get_default_metas

from .. import ZeroShotTFClassifier


@pytest.fixture(scope='function')
def test_metas(tmp_path):
    metas = get_default_metas()
    metas['workspace'] = str(tmp_path)
    if 'JINA_TEST_GPU' in os.environ:
        metas['on_gpu'] = True
    yield metas


_models = [
    'distilbert-base-uncased',
    'bert-base-uncased',
    'roberta-base',
    'xlnet-base-cased',
]


def _assert_params_equal(params_dict: dict, classifier: ZeroShotTFClassifier):
    for key, val in params_dict.items():
        assert val == getattr(classifier, key)


@pytest.mark.parametrize('model_name', _models)
@pytest.mark.parametrize('pooling_strategy', ['cls', 'mean', 'max'])
@pytest.mark.parametrize('layer_index', [-1, -2, 0])
def test_encoding_results(test_metas,
                          model_name,
                          pooling_strategy,
                          layer_index):
    params = {
        'pretrained_model_name_or_path': model_name,
        'pooling_strategy': pooling_strategy,
        'layer_index': layer_index,
    }

    test_data = np.array(['it is a good day!',
                          'the dog sits on the floor.'])
    test_labels = ['bla', 'bleep', 'bloop']

    classifier = ZeroShotTFClassifier(labels=test_labels,
                                      metas=test_metas, **params)

    encoded_data = classifier._encode(test_data)

    hidden_dim_sizes = {
        'distilbert-base-uncased': 768,
        'bert-base-uncased': 768,
        'roberta-base': 768,
        'xlm-roberta-base': 768,
        'xlnet-base-cased': 768,
    }
    hidden_dim_size = hidden_dim_sizes[
        classifier.pretrained_model_name_or_path
    ]
    assert encoded_data.shape == (2, hidden_dim_size)

    if classifier.pooling_strategy != 'cls' or \
            classifier.layer_index != 0:
        assert not np.allclose(encoded_data[0],
                               encoded_data[1],
                               rtol=1)
    else:
        assert np.allclose(encoded_data[0],
                           encoded_data[1],
                           atol=1e-5,
                           rtol=1e-4)


@pytest.mark.parametrize('model_name', ['bert-base-uncased'])
@pytest.mark.parametrize('pooling_strategy', ['cls', 'mean', 'max'])
@pytest.mark.parametrize('layer_index', [-1, -2])
def test_embedding_consistency(test_metas,
                               model_name,
                               pooling_strategy,
                               layer_index):
    params = {
        'pretrained_model_name_or_path': model_name,
        'pooling_strategy': pooling_strategy,
        'layer_index': layer_index,
    }
    test_data = np.array(['it is a good day!',
                          'the dog sits on the floor.'])

    test_labels = ['bla', 'bleep', 'bloop']

    classifier = ZeroShotTFClassifier(labels=test_labels,
                                      metas=test_metas, **params)
    encoded_data = classifier._encode(test_data)

    encoded_data_file = \
        f'tests/{model_name}-{pooling_strategy}-{layer_index}.npy'
    enc_data_loaded = np.load(encoded_data_file)

    np.testing.assert_allclose(encoded_data,
                               enc_data_loaded,
                               atol=1e-5,
                               rtol=1e-6)


@pytest.mark.parametrize('model_name', _models)
@pytest.mark.parametrize('pooling_strategy', ['cls', 'mean', 'max'])
@pytest.mark.parametrize('layer_index', [-1])
def test_max_length_truncation(test_metas,
                               model_name,
                               pooling_strategy,
                               layer_index):
    params = {
        'pretrained_model_name_or_path': model_name,
        'pooling_strategy': pooling_strategy,
        'layer_index': layer_index,
        'max_length': 3,
    }
    test_data = np.array(['it is a very good day!', 'it is a very sunny day!'])
    test_labels = ['bla', 'bleep', 'bloop']

    classifier = ZeroShotTFClassifier(labels=test_labels,
                                      metas=test_metas, **params)
    encoded_data = classifier._encode(test_data)

    np.testing.assert_allclose(encoded_data[0],
                               encoded_data[1],
                               atol=1e-5,
                               rtol=1e-4)


@pytest.mark.parametrize('model_name', _models)
@pytest.mark.parametrize('pooling_strategy', ['cls', 'mean', 'max'])
@pytest.mark.parametrize('layer_index', [-1, -2])
def test_shape_single_document(test_metas,
                               model_name,
                               pooling_strategy,
                               layer_index):
    params = {
        'pretrained_model_name_or_path': model_name,
        'pooling_strategy': pooling_strategy,
        'layer_index': layer_index,
        'max_length': 3,
    }
    test_data = np.array(['it is a very good day!'])
    test_labels = ['bla', 'bleep', 'bloop']

    classifier = ZeroShotTFClassifier(labels=test_labels,
                                      metas=test_metas, **params)
    encoded_data = classifier._encode(test_data)
    assert len(encoded_data.shape) == 2
    assert encoded_data.shape[0] == 1


@pytest.mark.parametrize('model_name', _models)
@pytest.mark.parametrize('pooling_strategy', ['min'])
@pytest.mark.parametrize('layer_index', [-2])
def test_save_and_load(test_metas, model_name, pooling_strategy, layer_index):
    params = {
        'pretrained_model_name_or_path': model_name,
        'pooling_strategy': pooling_strategy,
        'layer_index': layer_index,
    }
    test_labels = ['bla', 'bleep', 'bloop']

    classifier = ZeroShotTFClassifier(labels=test_labels,
                                      metas=test_metas, **params)

    classifier.save_config()
    _assert_params_equal(params, classifier)
    assert os.path.exists(classifier.config_abspath)

    test_data = np.array(['a', 'b', 'c', 'x', '!'])
    encoded_data_control = classifier._encode(test_data)

    classifier.touch()
    classifier.save()
    assert os.path.exists(classifier.save_abspath)

    classifier_loaded = BaseExecutor.load(classifier.save_abspath)
    _assert_params_equal(params, classifier_loaded)

    encoded_data_test = classifier_loaded._encode(test_data)
    np.testing.assert_array_equal(encoded_data_control, encoded_data_test)


@pytest.mark.parametrize('model_name', _models)
@pytest.mark.parametrize('pooling_strategy', ['min'])
@pytest.mark.parametrize('layer_index', [-2])
def test_save_and_load_config(test_metas,
                              model_name,
                              pooling_strategy,
                              layer_index):
    params = {
        'pretrained_model_name_or_path': model_name,
        'pooling_strategy': pooling_strategy,
        'layer_index': layer_index,
    }
    test_labels = ['bla', 'bleep', 'bloop']

    classifier = ZeroShotTFClassifier(labels=test_labels,
                                      metas=test_metas, **params)

    classifier.save_config()
    _assert_params_equal(params, classifier)
    assert os.path.exists(classifier.config_abspath)

    classifier_loaded = BaseExecutor.load_config(classifier.config_abspath)
    _assert_params_equal(params, classifier_loaded)


@pytest.mark.parametrize('layer_index', [-100, 100])
def test_wrong_layer_index(test_metas, layer_index):
    params = {'layer_index': layer_index}
    test_labels = ['bla', 'bleep', 'bloop']

    with pytest.raises(ValueError):
        ZeroShotTFClassifier(labels=test_labels,
                             metas=test_metas, **params)


def test_wrong_pooling_strategy():
    test_labels = ['bla', 'bleep', 'bloop']
    with pytest.raises(NotImplementedError):
        ZeroShotTFClassifier(labels=test_labels,
                             pooling_strategy='wrong')


@pytest.mark.parametrize(
    'params',
    [{'pooling_strategy': 'cls', 'pretrained_model_name_or_path': 'gpt2'}],
)
def test_no_cls_token(test_metas, params):
    test_labels = ['bla', 'bleep', 'bloop']

    with pytest.raises(ValueError):
        ZeroShotTFClassifier(labels=test_labels,
                             metas=test_metas, **params)


def test_classifier_prediction_type():
    test_data = np.array(['business', 'politics', 'food'])
    test_labels = ['business', 'politics', 'food']

    classifier = ZeroShotTFClassifier(labels=test_labels)

    predictions = classifier.predict(test_data)

    assert isinstance(predictions, np.ndarray)


@pytest.mark.parametrize('model_name', _models)
@pytest.mark.parametrize('pooling_strategy', ['min'])
@pytest.mark.parametrize('layer_index', [-2])
def test_classifier_prediction_results(test_metas,
                                       model_name,
                                       pooling_strategy,
                                       layer_index):
    params = {
        'pretrained_model_name_or_path': model_name,
        'pooling_strategy': pooling_strategy,
        'layer_index': layer_index,
    }

    test_data = np.array(['business', 'politics', 'food'])
    test_labels = ['business', 'politics', 'food']

    classifier = ZeroShotTFClassifier(labels=test_labels,
                                      metas=test_metas, **params)

    predictions = classifier.predict(test_data)

    predictions_expected = np.array([[1, 0, 0],
                                     [0, 1, 0],
                                     [0, 0, 1]])

    np.testing.assert_array_equal(predictions, predictions_expected)


def test_classifier_predictions_shape():
    test_data = np.array(['it is a good day!', 'the dog sits on the floor.'])
    test_labels = ['business', 'movie', 'food']

    classifier = ZeroShotTFClassifier(labels=test_labels)
    predictions = classifier.predict(test_data)

    assert test_data.shape[0] == predictions.shape[0]
    assert len(test_labels) == predictions.shape[1]
    assert predictions.sum() == test_data.shape[0]


def test_classifier_evaluation():
    test_labels = ['bla', 'bleep', 'bloop']

    classifier = ZeroShotTFClassifier(labels=test_labels)

    assert pytest.approx(classifier._evaluate([1, 1, 1], [1, 1, 1])) == 0
    assert pytest.approx(classifier._evaluate([1, 0, 1], [0, 1, 0])) == 1


def test_classifier_invalid_labels():
    test_labels = ['business']

    with pytest.raises(ValueError):
        ZeroShotTFClassifier(labels=test_labels)


def test_classifier_missing_labels():
    with pytest.raises(TypeError):
        ZeroShotTFClassifier()


def test_classifier_duplicate_labels():
    test_labels = ['business', 'business', 'business']

    with pytest.raises(ValueError):
        ZeroShotTFClassifier(labels=test_labels)
