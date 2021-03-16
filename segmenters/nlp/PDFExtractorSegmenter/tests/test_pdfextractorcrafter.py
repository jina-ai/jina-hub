import os
import pytest

import numpy as np
from PIL import Image

from .. import PDFExtractorSegmenter

expected_text = "A cat poem\nI love cats, I love every kind of cat,\nI just wanna hug all of them, but I can't," \
                "\nI'm thinking about cats again\nI think about how cute they are\nAnd their whiskers and their " \
                "nose\n"

cur_dir = os.path.dirname(os.path.abspath(__file__))
path_img_text = os.path.join(cur_dir, 'cats_are_awesome.pdf')
path_text = os.path.join(cur_dir, 'cats_are_awesome_text.pdf')
path_img = os.path.join(cur_dir, 'cats_are_awesome_img.pdf')

with open(path_text, 'rb') as pdf:
    input_bytes_text = pdf.read()

with open(path_img, 'rb') as pdf:
    input_bytes_image = open(path_img, 'rb').read()

with open(path_img_text, 'rb') as pdf:
    input_bytes_images_text = pdf.read()


@pytest.mark.parametrize('inputs', [
    [np.stack([path_img_text, path_img_text]), [None, None]],
    [[None, None], [input_bytes_images_text, input_bytes_images_text]],
])
def test_io_images_and_text(inputs):
    segmenter = PDFExtractorSegmenter()
    docs_chunks = segmenter.segment(*inputs)
    assert len(docs_chunks) == 2
    for chunks in docs_chunks:

        assert len(chunks) == 3

        # Check images
        for idx, c in enumerate(chunks[:-1]):
            with Image.open(os.path.join(cur_dir, f'test_img_{idx}.jpg')) as img:
                blob = chunks[idx]['blob']
                assert chunks[idx]['mime_type'] == 'image/png'
                assert blob.shape[1], blob.shape[0] == img.size
                if idx == 0:
                    assert blob.shape == (660, 1024, 3)
                if idx == 1:
                    assert blob.shape == (626, 1191, 3)

        # Check text
        assert chunks[2]['text'] == expected_text
        assert chunks[2]['mime_type'] == 'text/plain'


@pytest.mark.parametrize('inputs', [
    [np.stack([path_text, path_text]), [None, None]],
    [[None, None], [input_bytes_text, input_bytes_text]],
])
def test_io_text(inputs):
    segmenter = PDFExtractorSegmenter()
    docs_chunks = segmenter.segment(*inputs)
    assert len(docs_chunks) == 2
    for chunks in docs_chunks:
        assert len(chunks) == 1
        # Check test
        assert chunks[0]['text'] == expected_text
        assert chunks[0]['mime_type'] == 'text/plain'


@pytest.mark.parametrize('inputs', [
    [np.stack([path_img, path_img]), [None, None]],
    [[None, None], [input_bytes_image, input_bytes_image]],
])
def test_io_img(inputs):
    segmenter = PDFExtractorSegmenter()
    docs_chunks = segmenter.segment(*inputs)
    assert len(docs_chunks) == 2
    for chunks in docs_chunks:
        assert len(chunks) == 2
        # Check images
        for idx, c in enumerate(chunks):
            with Image.open(os.path.join(cur_dir, f'test_img_{idx}.jpg')) as im:
                blob = chunks[idx]['blob']
                assert chunks[idx]['mime_type'] == 'image/png'
                assert blob.shape[1], blob.shape[0] == img.size
                if idx == 0:
                    assert blob.shape == (660, 1024, 3)
                if idx == 1:
                    assert blob.shape == (626, 1191, 3)
