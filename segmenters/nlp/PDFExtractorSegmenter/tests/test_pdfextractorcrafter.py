from .. import PDFExtractorSegmenter
from PIL import Image
import os

expected_text = "A cat poem\nI love cats, I love every kind of cat,\nI just wanna hug all of them, but I can't," \
                "\nI'm thinking about cats again\nI think about how cute they are\nAnd their whiskers and their " \
                "nose\n"

cur_dir = os.path.dirname(os.path.abspath(__file__))
path_img_text = os.path.join(cur_dir, 'cats_are_awesome.pdf')
path_text = os.path.join(cur_dir, 'cats_are_awesome_text.pdf')
path_img = os.path.join(cur_dir, 'cats_are_awesome_img.pdf')


def test_io_uri_images_and_text():
    crafter = PDFExtractorSegmenter()
    chunks = crafter.segment(uri=path_img_text, buffer=None)

    assert len(chunks) == 3

    # Check images
    for idx, c in enumerate(chunks[:-1]):
        img = Image.open(os.path.join(cur_dir, f'test_img_{idx}.jpg'))
        blob = chunks[idx]['blob']
        assert chunks[idx]['mime_type'] == 'image/png'
        assert blob.shape[1], blob.shape[0] == img.size
        if idx == 0:
            assert blob.shape == (660, 1024, 3)
        if idx == 1:
            assert blob.shape == (626, 1191, 3)

    # Check text
    assert chunks[2]['text'] == expected_text
    assert chunks[2]['mime_type'] == "text/plain"


def test_io_uri_text():
    crafter = PDFExtractorSegmenter()
    chunks = crafter.segment(uri=path_text, buffer=None)

    assert len(chunks) == 1

    # Check test
    assert chunks[0]['text'] == expected_text
    assert chunks[0]['mime_type'] == 'text/plain'


def test_io_uri_img():
    crafter = PDFExtractorSegmenter()
    chunks = crafter.segment(uri=path_img, buffer=None)

    assert len(chunks) == 2

    # Check images
    for idx, c in enumerate(chunks):
        img = Image.open(os.path.join(cur_dir, f'test_img_{idx}.jpg'))
        blob = chunks[idx]['blob']
        assert chunks[idx]['mime_type'] == 'image/png'
        assert blob.shape[1], blob.shape[0] == img.size
        if idx == 0:
            assert blob.shape == (660, 1024, 3)
        if idx == 1:
            assert blob.shape == (626, 1191, 3)


def test_io_buffer_images_and_text():
    with open(path_img_text, 'rb') as pdf:
        input_bytes = pdf.read()
    crafter = PDFExtractorSegmenter()
    chunks = crafter.segment(uri=None, buffer=input_bytes)

    assert len(chunks) == 3

    # Check images
    for idx, c in enumerate(chunks[:-1]):
        img = Image.open(os.path.join(cur_dir, f'test_img_{idx}.jpg'))
        blob = chunks[idx]['blob']
        assert chunks[idx]['mime_type'] == 'image/png'
        assert blob.shape[1], blob.shape[0] == img.size
        if idx == 0:
            assert blob.shape == (660, 1024, 3)
        if idx == 1:
            assert blob.shape == (626, 1191, 3)

    # Check test
    assert chunks[2]['text'] == expected_text
    assert chunks[2]['mime_type'] == 'text/plain'


def test_io_buffer_text():
    with open(path_text, 'rb') as pdf:
        input_bytes = pdf.read()
    crafter = PDFExtractorSegmenter()
    chunks = crafter.segment(uri=None, buffer=input_bytes)

    assert len(chunks) == 1

    # Check test
    assert chunks[0]['text'] == expected_text
    assert chunks[0]['mime_type'] == 'text/plain'


def test_io_buffer_img():
    with open(path_img, 'rb') as pdf:
        input_bytes = pdf.read()
    crafter = PDFExtractorSegmenter()
    chunks = crafter.segment(uri=None, buffer=input_bytes)

    assert len(chunks) == 2

    # Check images
    for idx, c in enumerate(chunks):
        img = Image.open(os.path.join(cur_dir, f'test_img_{idx}.jpg'))
        blob = chunks[idx]['blob']
        assert chunks[idx]['mime_type'] == 'image/png'
        assert blob.shape[1], blob.shape[0] == img.size
        if idx == 0:
            assert blob.shape == (660, 1024, 3)
        if idx == 1:
            assert blob.shape == (626, 1191, 3)
