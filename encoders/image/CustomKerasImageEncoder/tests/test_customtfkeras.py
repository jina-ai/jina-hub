import tempfile
import os

import numpy as np

from .. import CustomKerasImageEncoder
from jina.executors.metas import get_default_metas
from jina.executors import BaseExecutor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Activation, Flatten, Dense


def rm_files(file_paths):
    for file_path in file_paths:
        if os.path.exists(file_path):
            if os.path.isfile(file_path):
                os.remove(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path, ignore_errors=False, onerror=None)

class TestNet:
    def __init__(self):
        self.model = None
        self.input_shape = (224, 224, 3)
        self.conv = Conv2D(32, (3, 3), padding='same', name='conv1', input_shape=self.input_shape)
        self.activation_relu = Activation('relu')
        self.flatten = Flatten()
        self.dense = Dense(10, name='dense')
        self.activation_softmax = Activation('softmax')

    def create_model(self):
        self.model = Sequential()
        self.model.add(self.conv)
        self.model.add(self.activation_relu)
        self.model.add(self.flatten)
        self.model.add(self.dense)
        self.model.add(self.activation_softmax)
        return self.model

def get_encoder():
    metas = get_default_metas()
    if 'JINA_TEST_GPU' in os.environ:
        metas['on_gpu'] = True
    path = tempfile.NamedTemporaryFile().name
    model = TestNet().create_model()
    model.save(path)
    return CustomKerasImageEncoder(channel_axis=1, model_path=path, layer_name='dense', metas=metas)

def test_encoding_results():
    target_output_dim = 10
    input_dim = 224
    encoder = get_encoder()
    test_data = np.random.rand(2, 3, input_dim, input_dim)
    encoded_data = encoder.encode(test_data)
    assert encoded_data.shape == (2, target_output_dim)

def test_save_and_load():
    input_dim = 224
    encoder = get_encoder()
    test_data = np.random.rand(2, 3, input_dim, input_dim)
    encoded_data_control = encoder.encode(test_data)
    encoder.touch()
    encoder.save()
    assert os.path.exists(encoder.save_abspath)
    encoder_loaded = BaseExecutor.load(encoder.save_abspath)
    encoded_data_test = encoder_loaded.encode(test_data)
    assert encoder_loaded.channel_axis == encoder.channel_axis
    np.testing.assert_array_equal(encoded_data_control, encoded_data_test)

def test_save_and_load_config():
    encoder = get_encoder()
    encoder.save_config()
    assert os.path.exists(encoder.config_abspath)
    encoder_loaded = BaseExecutor.load_config(encoder.config_abspath)
    assert encoder_loaded.channel_axis == encoder.channel_axis