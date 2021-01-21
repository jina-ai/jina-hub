import os

import pytest
import numpy as np

from jina.executors.metas import get_default_metas
from jina.executors import BaseExecutor

from .. import ImageOnnxEncoder


@pytest.fixture
def encoder(tmpdir):
    model_path = 'models/vision/classification/mobilenet/model/mobilenetv2-7.onnx'

    metas = get_default_metas()
    if 'JINA_TEST_GPU' in os.environ:
        metas['on_gpu'] = True
    metas['workspace'] = str(tmpdir)
    return ImageOnnxEncoder(output_feature='mobilenetv20_features_relu1_fwd',
                            model_path=model_path, metas=metas)


input_dim = 224
output_dim = 1280
num_samples = 2


def test_encoding_results(encoder):
    test_data = np.random.rand(num_samples, 3, input_dim, input_dim)
    encoded_data = encoder.encode(test_data)
    assert encoded_data.shape == (num_samples, output_dim)


def test_save_and_load(encoder):
    test_data = np.random.rand(num_samples, 3, input_dim, input_dim)
    encoded_data_control = encoder.encode(test_data)
    encoder.touch()
    encoder.save()
    assert os.path.exists(encoder.save_abspath)
    encoder_loaded = BaseExecutor.load(encoder.save_abspath)
    encoded_data_test = encoder_loaded.encode(test_data)
    assert encoder_loaded.raw_model_path == encoder.raw_model_path
    np.testing.assert_array_equal(encoded_data_control, encoded_data_test)


def test_save_and_load_config(encoder):
    encoder.save_config()
    assert os.path.exists(encoder.config_abspath)
    encoder_loaded = BaseExecutor.load_config(encoder.config_abspath)
    assert encoder_loaded.raw_model_path == encoder.raw_model_path
