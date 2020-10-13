import os
import numpy as np
import pytest
from jina.executors.metas import get_default_metas
from jina.executors import BaseExecutor
from .. import ImagePaddlehubEncoder

input_dim = 224
target_output_dim = 2048
num_doc = 2
test_data = np.random.rand(num_doc, 3, input_dim, input_dim)
tmp_files = []


@pytest.fixture(scope='function', autouse=True)
def metas(tmpdir):
    metas = get_default_metas()
    if 'JINA_TEST_GPU' in os.environ:
        metas['on_gpu'] = True
    metas['workspace'] = str(tmpdir)
    yield metas


def test_imagepaddlehubencoder_encode(metas):
    encoder = ImagePaddlehubEncoder(metas=metas)
    encoded_data = encoder.encode(test_data)
    assert encoded_data.shape == (num_doc, target_output_dim)


def test_imagepaddlehubencoder_save_and_load(metas):
    encoder = ImagePaddlehubEncoder(metas=metas)
    encoder.touch()
    encoder.save()
    assert os.path.exists(encoder.save_abspath)
    encoder_loaded = BaseExecutor.load(encoder.save_abspath)
    assert encoder_loaded.model_name == encoder.model_name


def test_imagepaddlehubencoder_save_and_load_config(metas):
    encoder = ImagePaddlehubEncoder(metas=metas)
    encoder.save_config()
    assert os.path.exists(encoder.config_abspath)
    encoder_loaded = BaseExecutor.load_config(encoder.config_abspath)
    assert encoder_loaded.model_name == encoder.model_name
