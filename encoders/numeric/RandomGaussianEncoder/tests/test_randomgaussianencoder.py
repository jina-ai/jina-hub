import os
import shutil
import numpy as np
import pickle

from .. import RandomGaussianEncoder
from .. import TransformEncoder

workspace = os.path.join(os.environ['TEST_WORKDIR'], 'test_tmp')



def get_encoder(encoder):
    assert encoder is not None
    if encoder is not None:
        encoder.workspace = workspace
        assert os.path.exists(encoder.index_abspath)
        tmp_files = encoder.save_abspath
    return encoder


def rm_files(file_paths):
    for file_path in file_paths:
        if os.path.exists(file_path):
            if os.path.isfile(file_path):
                os.remove(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path, ignore_errors=False, onerror=None)


def test_randomgaussianencodertrain(self):
    input_dim = 28
    target_output_dim = 7
    encoder = get_encoder(RandomGaussianEncoder(output_dim=target_output_dim))
    train_data = np.random.rand(2000, input_dim)
    encoded_data = encoder.encode(train_data)
    self.assertEqual(encoded_data.shape, (train_data.shape[0], self.target_output_dim))
    self.assertIs(type(encoded_data), np.ndarray)
    encoder.train(train_data)


def test_randomgaussianencoderload(self):
    input_dim = 28
    target_output_dim = 2
    encoder = get_encoder(RandomGaussianEncoder(output_dim=target_output_dim))
    train_data = np.random.rand(2000, input_dim)
    encoder.train(train_data)
    filename = 'random_gaussian_model.model'
    assert os.path.exists(filename.index_abspath)
    index_abspath = filename.index_abspath
    pickle.dump(encoder.model, open(filename, 'wb'))
    rm_files([index_abspath])
