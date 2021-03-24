import os
from pathlib import Path

import pytest

from PIL import Image
import numpy as np

from .. import FaceNetSegmenter


SEGMENTER_DIR = Path(__file__).parent.parent


@pytest.fixture
def segmenter():
    return FaceNetSegmenter()


def assert_correct_output(result, n_faces: int):
    assert isinstance(result, list)
    assert len(result) == n_faces
    for face in result:
        assert face['blob'].shape == (3, 160, 160)


@pytest.mark.parametrize('filename', ['one_face.jpg',
                                      'three_faces.jpg',
                                      'four_faces.jpg',
                                      'five_faces.jpg'])
def test_segment_face(segmenter, filename):
    image = Image.open(SEGMENTER_DIR / 'imgs' / filename)

    result = segmenter.segment(np.array(image))

    assert_correct_output(result, n_faces=1)


@pytest.mark.parametrize('height', [128, 512])
@pytest.mark.parametrize('width', [128, 512])
def test_segment_no_face(segmenter, height: int, width: int):
    image = np.zeros((width, height, 3))

    result = segmenter.segment(image)

    assert_correct_output(result, n_faces=0)


@pytest.fixture
def segmenter_multiface():
    return FaceNetSegmenter(keep_all=True, post_process=False)


@pytest.mark.parametrize('filename, n_faces', [
    ('three_faces.jpg', 3),
    ('four_faces.jpg', 4),
    ('five_faces.jpg', 5)
])
def test_segment_face(segmenter_multiface, filename, n_faces):
    image = Image.open(os.path.join(SEGMENTER_DIR, 'imgs', filename))

    result = segmenter_multiface.segment(np.array(image))

    assert_correct_output(result, n_faces=n_faces)


if __name__ == '__main__':
    pytest.main()
