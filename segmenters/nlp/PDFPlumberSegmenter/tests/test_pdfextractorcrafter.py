import os

import pytest
from PIL import Image

from .. import PDFPlumberSegmenter

CUR_DIR = os.path.dirname(os.path.abspath(__file__))


@pytest.fixture
def expected_text():
    expected_text = "A cat poem\nI love cats, I love every kind of cat,\nI just wanna hug all of them, but I can't," \
                    "\nI'm thinking about cats again\nI think about how cute they are\nAnd their whiskers and their " \
                    "nose"
    return expected_text


@pytest.fixture
def input_pdf():
    path_img_text = os.path.join(CUR_DIR, 'cats_are_awesome.pdf')
    path_text = os.path.join(CUR_DIR, 'cats_are_awesome_text.pdf')
    path_img = os.path.join(CUR_DIR, 'cats_are_awesome_img.pdf')

    with open(path_text, 'rb') as pdf:
        input_bytes_text = pdf.read()

    with open(path_img, 'rb') as pdf:
        input_bytes_image = pdf.read()

    with open(path_img_text, 'rb') as pdf:
        input_bytes_images_text = pdf.read()

    return {'img_text': [(path_img_text, None), (None, input_bytes_images_text)],
            'text': [(path_text, None), (None, input_bytes_text)],
            'img': [(path_img, None), (None, input_bytes_image)]}


def test_io_images_and_text(input_pdf, expected_text):
    segmenter = PDFPlumberSegmenter()
    for uri, buffer in input_pdf['img_text']:
        chunks = segmenter.segment(uri, buffer, 'application/pdf')
        assert len(chunks) == 4
        # Check images
        for idx, c in enumerate(chunks[:2]):
            with Image.open(os.path.join(CUR_DIR, f'test_img_{idx}.jpg')) as img:
                blob = chunks[idx]['blob']
                assert chunks[idx]['mime_type'] == 'image/png'
                assert blob.shape[1], blob.shape[0] == img.size
                if idx == 0:
                    assert blob.shape == (660, 1024, 3)
                if idx == 1:
                    assert blob.shape == (626, 1191, 3)

            # Check text
            assert chunks[2]['text'] == 'A cat poem'
            assert chunks[2]['tags']['title']
            assert chunks[2]['mime_type'] == 'text/plain'
            assert chunks[3]['text'] == expected_text
            assert chunks[3]['mime_type'] == 'text/plain'


def test_io_text(input_pdf, expected_text):
    segmenter = PDFPlumberSegmenter()
    for uri, buffer in input_pdf['text']:
        chunks = segmenter.segment(uri, buffer, 'application/pdf')
        assert len(chunks) == 2
        # Check test
        assert chunks[0]['text'] == 'A cat poem'
        assert chunks[0]['tags']['title']
        assert chunks[0]['mime_type'] == 'text/plain'
        assert chunks[1]['text'] == expected_text
        assert chunks[1]['mime_type'] == 'text/plain'


def test_io_img(input_pdf):
    segmenter = PDFPlumberSegmenter()
    for uri, buffer in input_pdf['img']:
        chunks = segmenter.segment(uri, buffer, 'application/pdf')
        assert len(chunks) == 2
        # Check images
        for idx, c in enumerate(chunks):
            with Image.open(os.path.join(CUR_DIR, f'test_img_{idx}.jpg')) as img:
                blob = chunks[idx]['blob']
                assert chunks[idx]['mime_type'] == 'image/png'
                assert blob.shape[1], blob.shape[0] == img.size
                if idx == 0:
                    assert blob.shape == (660, 1024, 3)
                if idx == 1:
                    assert blob.shape == (626, 1191, 3)
