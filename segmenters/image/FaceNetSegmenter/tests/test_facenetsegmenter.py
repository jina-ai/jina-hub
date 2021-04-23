import os
from pathlib import Path
from typing import List
import logging

import pytest

from PIL import Image
import cv2
import numpy as np

from .. import FaceNetSegmenter


SEGMENTER_DIR = Path(__file__).parent.parent


@pytest.fixture
def segmenter():
    return FaceNetSegmenter()


def assert_correct_output(result, n_faces: int):
    #assert isinstance(result, list)
    assert len(result) == n_faces
    for face in result:
        assert face['blob'].shape == (3, 160, 160)

@pytest.mark.parametrize('filenames_with_labels', [
    [('one_face.jpg', 1)],
    [('three_faces.jpg', 3), ('four_faces.jpg', 4)],
    [('one_face.jpg', 1), ('three_faces.jpg', 3), ('four_faces.jpg', 4), ('five_faces.jpg',5)],
])
def test_segment_face_in_batch(segmenter_multiface, filenames_with_labels):
    images = [
        cv2.resize(np.array(Image.open(os.path.join(SEGMENTER_DIR, 'imgs', filename[0]))),dsize=(480, 320))
        for filename in filenames_with_labels
    ]

    results = segmenter_multiface.segment(images) #[[dict1,dict2...],...] each image gets a list of dicts, which relates to the faces
    i = 0
    for result in results:
        assert_correct_output(result, n_faces=filenames_with_labels[i][1])
        i += 1


@pytest.mark.parametrize('height', [128, 512])
@pytest.mark.parametrize('width', [128, 512])
@pytest.mark.parametrize('batch_size', [1, 4, 16])
def test_segment_no_face(segmenter, height: int, width: int, batch_size: int):
    images = np.zeros((batch_size, width, height, 3))

    results = segmenter.segment(images)

    for result in results:
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
    result = segmenter_multiface.segment([np.array(image)])[0]

    assert_correct_output(result, n_faces=n_faces)


if __name__ == '__main__':
    pytest.main()
