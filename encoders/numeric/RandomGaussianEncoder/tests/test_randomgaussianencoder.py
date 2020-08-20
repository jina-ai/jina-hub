import os
import shutil
import numpy as np
import pickle

from .. import RandomGaussianEncoder
from .. import TransformEncoder

def rm_files(file_paths):
    for file_path in file_paths:
        if os.path.exists(file_path):
            if os.path.isfile(file_path):
                os.remove(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path, ignore_errors=False, onerror=None)

def test_randomgaussinencodertrain():
    requires_train_after_load = True
    input_dim = 28
    target_output_dim = 7
    encoder = RandomGaussianEncoder(output_dim=target_output_dim)
    train_data = np.random.rand(2000, input_dim)
    encoder.train(train_data)


def test_randomgaussinencoderload():
    requires_train_after_load = False
    input_dim = 28
    target_output_dim = 2
    encoder = RandomGaussianEncoder(output_dim=target_output_dim)
    train_data = np.random.rand(2000, input_dim)
    encoder.train(train_data)
    filename = 'random_gaussian_model.model'

    assert os.path.exists(filename.index_abspath)

    index_abspath = filename.index_abspath
    pickle.dump(encoder.model, open(filename, 'wb'))

    rm_files([index_abspath])



