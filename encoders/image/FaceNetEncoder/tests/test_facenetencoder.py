import os
from pathlib import Path

from PIL import Image
import pytest
import numpy as np

from .. import FaceNetEncoder


ENCODER_DIR = Path(__file__).parent.parent


@pytest.fixture
def encoder():
    return FaceNetEncoder()


@pytest.mark.parametrize('batch_size', [1, 2, 32])
def test_encoding_face_batch(encoder, batch_size: int):
    image = Image.open(os.path.join(ENCODER_DIR, 'imgs', 'man_piercing.jpg'))
    image = np.array(image).transpose((2, 0, 1))  # channels-first
    image = image / 128. - 127.5
    images = np.stack([image] * batch_size)

    result = encoder.encode(images)

    assert result.shape == (batch_size, 512)
    assert isinstance(result, np.ndarray)


@pytest.mark.parametrize('batch_size', [1, 8])
@pytest.mark.parametrize('height', [128, 512])
@pytest.mark.parametrize('width', [128, 512])
def test_encoding_no_face_batch(encoder, height: int, width: int, batch_size: int):
    image = np.zeros((3, width, height))
    images = np.stack([image] * batch_size)

    result = encoder.encode(images)

    assert result.shape == (batch_size, 512)
    assert isinstance(result, np.ndarray)


if __name__ == '__main__':
    pytest.main()
