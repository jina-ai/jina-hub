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
    chunks = crafter.craft(uri=path_img_text, buffer=None)

    assert len(chunks) == 3

    # Check images
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    img1 = Image.open(os.path.join(cur_dir, 'test_img_0.jpg'))
    img2 = Image.open(os.path.join(cur_dir, 'test_img_1.jpg'))

    blob1 = chunks[0]['blob']
    assert (blob1.shape[1], blob1.shape[0]) == img1.size

    blob2 = chunks[1]['blob']
    assert (blob2.shape[1], blob2.shape[0]) == img2.size

    # Check test
    assert chunks[2]['text'] == expected_text


def test_io_uri_text():
    crafter = PDFExtractorSegmenter()
    chunks = crafter.craft(uri=path_text, buffer=None)

    assert len(chunks) == 1

    # Check test
    assert chunks[0]['text'] == expected_text


def test_io_uri_img():
    crafter = PDFExtractorSegmenter()
    chunks = crafter.craft(uri=path_img, buffer=None)

    assert len(chunks) == 2

    # Check images
    for idx, c in enumerate(chunks):
        img = Image.open(os.path.join(cur_dir, f'test_img_{idx}.jpg'))
        blob = chunks[idx]['blob']
        assert blob.shape[1] == img.width
        blob.shape == img.size


def test_io_buffer_images_and_text():
    with open(path_img_text, 'rb') as pdf:
        input_bytes = pdf.read()
    crafter = PDFExtractorSegmenter()
    chunks = crafter.craft(uri=None, buffer=input_bytes)

    assert len(chunks) == 3

    # Check images
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    img1 = Image.open(os.path.join(cur_dir, 'test_img_0.jpg'))
    img2 = Image.open(os.path.join(cur_dir, 'test_img_1.jpg'))

    blob1 = chunks[0]['blob']
    assert (blob1.shape[1], blob1.shape[0]) == img1.size

    blob2 = chunks[1]['blob']
    assert (blob2.shape[1], blob2.shape[0]) == img2.size

    # Check test
    assert chunks[2]['text'] == expected_text


def test_io_buffer_text():
    with open(path_text, 'rb') as pdf:
        input_bytes = pdf.read()
    crafter = PDFExtractorSegmenter()
    chunks = crafter.craft(uri=None, buffer=input_bytes)

    assert len(chunks) == 1

    # Check test
    assert chunks[0]['text'] == expected_text


def test_io_buffer_img():
    with open(path_img, 'rb') as pdf:
        input_bytes = pdf.read()
    crafter = PDFExtractorSegmenter()
    chunks = crafter.craft(uri=None, buffer=input_bytes)

    assert len(chunks) == 2

    # Check images
    for idx, c in enumerate(chunks):
        img = Image.open(os.path.join(cur_dir, f'test_img_{idx}.jpg'))
        blob = chunks[idx]['blob']
        assert blob.shape[1] == img.width
        blob.shape == img.size
