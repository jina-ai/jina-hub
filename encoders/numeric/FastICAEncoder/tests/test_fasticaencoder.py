from .. import FastICAEncoder
from encoders.numeric import TransformEncoder
import os
import pickle
import numpy as np
import shutil

from jina.executors import BaseExecutor


input_dim = 28
target_output_dim = 2
train_data = np.random.rand(2000, input_dim)


def rm_files(tmp_files):
    for file in tmp_files:
        if file:
            if os.path.exists(file):
                if os.path.isfile(file):
                    os.remove(file)
                elif os.path.isdir(file):
                    shutil.rmtree(file, ignore_errors=False, onerror=None)


def get_encoder_TrainCase():
    encoder = FastICAEncoder(
        output_dim=target_output_dim, whiten=True, num_features=input_dim, max_iter=200)
    encoder.train(train_data)
    return encoder



def test_encoding_results_TrainCase():
    encoder = get_encoder_TrainCase()
    test_data = np.random.rand(10, input_dim)
    encoded_data = encoder.encode(test_data)
    assert encoded_data.shape == (test_data.shape[0], target_output_dim)
    assert type(encoded_data) is np.ndarray
    rm_files([encoder.save_abspath, encoder.config_abspath, encoder.model_path])

def test_save_and_load_TrainCase():
    encoder = get_encoder_TrainCase()
    encoder.touch()
    encoder.save()
    assert os.path.exists(encoder.save_abspath)
    rm_files([encoder.save_abspath, encoder.config_abspath, encoder.model_path])


def test_save_and_load_config_TrainCase():
    encoder = get_encoder_TrainCase()
    encoder.save_config()
    assert os.path.exists(encoder.config_abspath)
    encoder_loaded = BaseExecutor.load_config(encoder.config_abspath)
    encoder_loaded.train(np.random.rand(2000, input_dim))
    test_data = np.random.rand(10, input_dim)
    encoded_data_test = encoder_loaded.encode(test_data)
    assert encoded_data_test.shape == (10, target_output_dim)
    rm_files([encoder.save_abspath, encoder.config_abspath, encoder.model_path])


def get_encoder_LoadCase():
    encoder = FastICAEncoder(
        output_dim=target_output_dim, whiten=True, num_features=input_dim, max_iter=200)
    encoder.train(train_data)
    filename = 'ica_model.model'
    pickle.dump(encoder.model, open(filename, 'wb'))
    return TransformEncoder(model_path=filename)


def test_encoding_results_LoadCase():
    encoder = get_encoder_LoadCase()
    test_data = np.random.rand(10, input_dim)
    encoded_data = encoder.encode(test_data)
    assert encoded_data.shape == (test_data.shape[0], target_output_dim)
    assert type(encoded_data) is np.ndarray
    rm_files([encoder.save_abspath, encoder.config_abspath, encoder.model_path])

def test_save_and_load_LoadCase():
    encoder = get_encoder_LoadCase()
    test_data = np.random.rand(10, input_dim)
    encoded_data_control = encoder.encode(test_data)
    encoder.touch()
    encoder.save()
    assert os.path.exists(encoder.save_abspath)
    encoder_loaded = BaseExecutor.load(encoder.save_abspath)
    encoded_data_test = encoder_loaded.encode(test_data)
    # .all() to compare to vectors
    assert encoded_data_test.all() == encoded_data_control.all()
    rm_files([encoder.save_abspath, encoder.config_abspath, encoder.model_path])

def test_save_and_load_config_LoadCase():
    encoder = get_encoder_LoadCase()
    encoder.save_config()
    assert os.path.exists(encoder.config_abspath)
    encoder_loaded = BaseExecutor.load_config(encoder.config_abspath)
    test_data = np.random.rand(10, input_dim)
    encoded_data_test = encoder_loaded.encode(test_data)
    assert encoded_data_test.shape == (10, target_output_dim)
    rm_files([encoder.save_abspath, encoder.config_abspath, encoder.model_path])

