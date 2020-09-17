from .. import PDFExtractorSegmenter
from PIL import Image
import os

expected_text = "A cat poem\nI love cats, I love every kind of cat,\nI just wanna hug all of them, but I can't," \
                "\nI'm thinking about cats again\nI think about how cute they are\nAnd their whiskers and their " \
                "nose\n"


def test_io_uri_images_and_text():
    crafter = PDFExtractorSegmenter()
    chunks = crafter.craft(uri='cats_are_awesome.pdf', buffer=None)

    assert len(chunks) == 3

    # Check images
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    img1 = Image.open(os.path.join(cur_dir, '../test_img.jpg'))
    img2 = Image.open(os.path.join(cur_dir, '../test_img2.jpg'))

    blob1 = chunks[0]['blob']
    assert img1.width == blob1.width
    assert img1.height == blob1.height

    blob2 = chunks[1]['blob']
    assert img2.width == blob2.width
    assert img2.height == blob2.height

    # Check test
    assert chunks[2]['text'] == expected_text


def test_io_uri_text():
    crafter = PDFExtractorSegmenter()
    chunks = crafter.craft(uri='cats_are_awesome_text.pdf', buffer=None)

    assert len(chunks) == 1

    # Check test
    assert chunks[0]['text'] == expected_text


def test_io_uri_img():
    crafter = PDFExtractorSegmenter()
    chunks = crafter.craft(uri='cats_are_awesome_img.pdf', buffer=None)

    assert len(chunks) == 2

    # Check images
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    img1 = Image.open(os.path.join(cur_dir, '../test_img.jpg'))
    img2 = Image.open(os.path.join(cur_dir, '../test_img2.jpg'))

    blob1 = chunks[0]['blob']
    assert img1.width == blob1.width
    assert img1.height == blob1.height

    blob2 = chunks[1]['blob']
    assert img2.width == blob2.width
    assert img2.height == blob2.height


def test_io_buffer_images_and_text():
    with open('cats_are_awesome.pdf', 'rb') as pdf:
        input_bytes = pdf.read()
    crafter = PDFExtractorSegmenter()
    chunks = crafter.craft(uri=None, buffer=input_bytes)

    assert len(chunks) == 3

    # Check images
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    img1 = Image.open(os.path.join(cur_dir, '../test_img.jpg'))
    img2 = Image.open(os.path.join(cur_dir, '../test_img2.jpg'))

    blob1 = chunks[0]['blob']
    assert img1.width == blob1.width
    assert img1.height == blob1.height

    blob2 = chunks[1]['blob']
    assert img2.width == blob2.width
    assert img2.height == blob2.height

    # Check test
    assert chunks[2]['text'] == expected_text


def test_io_buffer_text():
    with open('cats_are_awesome_text.pdf', 'rb') as pdf:
        input_bytes = pdf.read()
    crafter = PDFExtractorSegmenter()
    chunks = crafter.craft(uri=None, buffer=input_bytes)

    assert len(chunks) == 1

    # Check test
    assert chunks[0]['text'] == expected_text


def test_io_buffer_img():
    with open('cats_are_awesome_img.pdf', 'rb') as pdf:
        input_bytes = pdf.read()
    crafter = PDFExtractorSegmenter()
    chunks = crafter.craft(uri=None, buffer=input_bytes)

    assert len(chunks) == 2

    # Check images
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    img1 = Image.open(os.path.join(cur_dir, '../test_img.jpg'))
    img2 = Image.open(os.path.join(cur_dir, '../test_img2.jpg'))

    blob1 = chunks[0]['blob']
    assert img1.width == blob1.width
    assert img1.height == blob1.height

    blob2 = chunks[1]['blob']
    assert img2.width == blob2.width
    assert img2.height == blob2.height
