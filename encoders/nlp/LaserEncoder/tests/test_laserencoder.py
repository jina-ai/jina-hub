import os
import shutil

import numpy as np
import pytest
from jina.executors import BaseExecutor
from jina.executors.metas import get_default_metas

from .. import LaserEncoder


def rm_files(tmp_files):
    for k in tmp_files:
        if os.path.exists(k):
            if os.path.isfile(k):
                os.remove(k)
            elif os.path.isdir(k):
                shutil.rmtree(k, ignore_errors=False, onerror=None)


os.environ["TEST_WORKDIR"] = os.getcwd()

metas = get_default_metas()
if "JINA_TEST_GPU" in os.environ:
    metas["on_gpu"] = True

encoders = [
    LaserEncoder(language="en", metas=metas),
    LaserEncoder(language="zh", metas=metas),
    LaserEncoder(language="ja", metas=metas),
]


@pytest.mark.parametrize("encoder", encoders)
def test_encoding_results(encoder):
    target_output_dim = 1024
    test_data = np.array(["it is a good day!", "the dog sits on the floor."])
    encoded_data = encoder.encode(test_data)
    assert encoded_data.shape == (2, target_output_dim)


@pytest.mark.parametrize("encoder", encoders)
def test_save_and_load(encoder):
    encoder.save_config()
    assert os.path.exists(encoder.config_abspath)
    test_data = np.array(["a", "b", "c", "x", "!"])
    encoded_data_control = encoder.encode(test_data)
    encoder.touch()
    encoder.save()
    assert os.path.exists(encoder.save_abspath)
    encoder_loaded = BaseExecutor.load(encoder.save_abspath)
    encoded_data_test = encoder_loaded.encode(test_data)
    np.testing.assert_array_equal(encoded_data_control, encoded_data_test)
    rm_files(
        [
            encoder.config_abspath,
            encoder.save_abspath,
            encoder_loaded.config_abspath,
            encoder_loaded.save_abspath,
        ]
    )


@pytest.mark.parametrize("encoder", encoders)
def test_save_and_load_config(encoder):
    encoder.save_config()
    assert os.path.exists(encoder.config_abspath)
    encoder_loaded = BaseExecutor.load_config(encoder.config_abspath)
    rm_files([encoder_loaded.config_abspath, encoder_loaded.save_abspath])
