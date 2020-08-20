import numpy as np
import pickle

from .. import RandomGaussianEncoder
from .. import TransformEncoder


class RandomGaussianEncoderLoadTestCase:
    def test_randomgaussinencoderload(self):
        requires_train_after_load = False
        input_dim = 28
        target_output_dim = 2
        encoder = RandomGaussianEncoder(output_dim=target_output_dim)
        train_data = np.random.rand(2000, input_dim)
        encoder.train(train_data)
        filename = 'random_gaussian_model.model'
        pickle.dump(encoder.model, open(filename, 'wb'))
        return TransformEncoder(model_path=filename)