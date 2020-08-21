import numpy as np
import pickle


from .. import FastICAEncoder
from encoders.numeric.test import TestClassNumeric
from encoders.numeric import TransformEncoder



class TestClassFastICATrain(TestClassNumeric):
    def _get_encoder(self):
        self.requires_train_after_load = True
        self.input_dim = 28
        self.target_output_dim = 2
        encoder = FastICAEncoder(
            output_dim=self.target_output_dim, whiten=True, num_features=self.input_dim, max_iter=200)
        self.train_data = np.random.rand(2000, self.input_dim)
        encoder.train(self.train_data)
        return encoder


class TestClassFastICALoad(TestClassNumeric):
    def _get_encoder(self):
        self.requires_train_after_load = False
        self.input_dim = 28
        self.target_output_dim = 2
        encoder = FastICAEncoder(
            output_dim=self.target_output_dim, whiten=True, num_features=self.input_dim, max_iter=200)
        self.train_data = np.random.rand(2000, self.input_dim)
        encoder.train(self.train_data)
        filename = 'ica_model.model'
        pickle.dump(encoder.model, open(filename, 'wb'))
        self.add_tmpfile(filename)
        return TransformEncoder(model_path=filename)
