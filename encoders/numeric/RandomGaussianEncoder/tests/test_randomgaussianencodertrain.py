import numpy as np

from .. import RandomGaussianEncoder


class RandomGaussianEncoderTrainTestCase:
    def test_randomgaussinencodertrain(self):
        requires_train_after_load = True
        input_dim = 28
        target_output_dim = 7
        encoder = RandomGaussianEncoder(output_dim=target_output_dim)
        train_data = np.random.rand(2000, input_dim)
        encoder.train(train_data)
        return encoder
