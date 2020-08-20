import pytest
import numpy as np

from jina.hub.encoders.image.torchvision import ImageTorchEncoder
from tests.unit.executors.encoders.image import ImageTestCase


class TorchVisionTestCase(ImageTestCase):
    def _get_encoder(self, metas):
        self.target_output_dim = 1280
        self.input_dim = 224
        return ImageTorchEncoder(metas=metas)

    def test_encoding_results(self):
        encoder = self.get_encoder()
        if encoder is None:
            return
        test_data = np.random.rand(2, 3, self.input_dim, self.input_dim)
        encoded_data = encoder.encode(test_data)
        assert encoded_data.shape == (2, self.target_output_dim)
