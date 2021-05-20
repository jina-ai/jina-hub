import os
from pathlib import Path
from typing import List

import numpy as np
import pytest
from PIL import Image


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


@pytest.mark.parametrize('filenames', [
    ['one_face.jpg'],
    ['three_faces.jpg', 'four_faces.jpg'],
    ['one_face.jpg', 'three_faces.jpg', 'four_faces.jpg', 'five_faces.jpg'],
])
def test_segment_face_single(segmenter, filenames: List[str]):
    images = [
        cv2.resize(
            np.array(Image.open(os.path.join(SEGMENTER_DIR, 'imgs', labelled_file))),
            dsize=(480, 320),
        )
        for labelled_file, _ in filenames_with_labels
    ]

    results = segmenter_multiface.segment(images)
    # [[dict1,dict2...],...] each image gets a list of dicts, which relates to the faces

    for result, (_, num_faces) in zip(results, filenames_with_labels):
        assert_correct_output(result, n_faces=num_faces)


@pytest.mark.parametrize(
    'filenames_with_labels',
    [
        [('one_face.jpg', 1), (np.zeros((320, 480, 3)), 0), ('three_faces.jpg', 3)],
    ],
)
def test_segment_mixed_batch(segmenter, filenames_with_labels):
    images = []
    for labelled_file, _ in filenames_with_labels:
        if type(labelled_file) == str:
            labelled_file = np.array(Image.open(os.path.join(SEGMENTER_DIR, 'imgs', labelled_file)))
            labelled_file = cv2.resize(labelled_file, dsize=(480, 320),)
            images.append(labelled_file)
        else:
            images.append(labelled_file)


    results = segmenter.segment(images)
    # [[dict1,dict2...],...] each image gets a list of dicts, which relates to the faces

    for result, (_, num_faces) in zip(results, filenames_with_labels):
        assert_correct_output(result, n_faces=num_faces)


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
def test_segment_face_multiple(segmenter_multiface, filename, n_faces):
    image = Image.open(os.path.join(SEGMENTER_DIR, 'imgs', filename))
    result = segmenter_multiface.segment([np.array(image)])[0]
    assert_correct_output(result, n_faces=n_faces)


@pytest.mark.parametrize('filename, n_faces, locations', [
    ('three_faces.jpg', 3, [[215.96368408203125, 108.82669830322266, 265.9007568359375, 172.01095581054688],
                            [139.0802459716797, 109.64990234375, 185.64785766601562, 174.12840270996094],
                            [274.59246826171875, 69.23793029785156, 325.1572265625, 126.35550689697266]])
])
def test_segment_face_location(segmenter_multiface, filename, n_faces, locations):
    image = Image.open(os.path.join(SEGMENTER_DIR, 'imgs', filename))
    result = segmenter_multiface.segment([np.array(image)])[0]

    for i in range(n_faces):
        assert len(result[i]['location']) == len(locations[i])


if __name__ == '__main__':
    pytest.main()
