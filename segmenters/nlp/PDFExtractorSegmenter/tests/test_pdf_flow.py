from jina import Document
from jina.flow import Flow
from PIL import Image
import os

cur_dir = os.path.dirname(os.path.abspath(__file__))

expected_text = "A cat poem\nI love cats, I love every kind of cat,\nI just wanna hug all of them, but I can't," \
                "\nI'm thinking about cats again\nI think about how cute they are\nAnd their whiskers and their " \
                "nose\n"


def validate_text_fn(resp):
    for d in resp.search.docs:
        assert expected_text == d.chunks[0].text


def validate_img_fn(resp):
    for d in resp.search.docs:
        for chunk in range(len(d.chunks)):
            img = Image.open(os.path.join(cur_dir, f'test_img_{chunk}.jpg'))
            blob = d.chunks[chunk].blob.dense
            assert blob.shape[1], blob.shape[0] == img.size


def validate_mix_fn(resp):
    for d in resp.search.docs:
        for chunk in range(len(d.chunks) - 1):
            img = Image.open(os.path.join(cur_dir, f'test_img_{chunk}.jpg'))
            blob = d.chunks[chunk].blob.dense
            assert blob.shape[1], blob.shape[0] == img.size

        assert expected_text == d.chunks[2].text


def search_generator(path: str, buffer: bytes):
    d = Document()
    if buffer:
        d.buffer = buffer
    if path:
        d.content = path
    yield d


def test_pdf_flow_text():
    path = os.path.join(cur_dir, 'cats_are_awesome_text.pdf')
    f = Flow().add(uses='PDFExtractorSegmenter', array_in_pb=True)
    with f:
        f.search(input_fn=search_generator(path=path, buffer=None), on_done=validate_text_fn)


def test_pdf_flow_img():
    path = os.path.join(cur_dir, 'cats_are_awesome_img.pdf')
    f = Flow().add(uses='PDFExtractorSegmenter', array_in_pb=True)
    with f:
        f.search(input_fn=search_generator(path=path, buffer=None), on_done=validate_img_fn)


def test_pdf_flow_mix():
    path = os.path.join(cur_dir, 'cats_are_awesome.pdf')
    f = Flow().add(uses='PDFExtractorSegmenter', array_in_pb=True)
    with f:
        f.search(input_fn=search_generator(path=path, buffer=None), on_done=validate_mix_fn)


def test_pdf_flow_text_buffer():
    path = os.path.join(cur_dir, 'cats_are_awesome_text.pdf')
    with open(path, 'rb') as pdf:
        input_bytes = pdf.read()
    f = Flow().add(uses='PDFExtractorSegmenter', array_in_pb=True)
    with f:
        f.search(input_fn=search_generator(path=None, buffer=input_bytes), on_done=validate_text_fn)


def test_pdf_flow_img_buffer():
    path = os.path.join(cur_dir, 'cats_are_awesome_img.pdf')
    with open(path, 'rb') as pdf:
        input_bytes = pdf.read()
    f = Flow().add(uses='PDFExtractorSegmenter', array_in_pb=True)
    with f:
        f.search(input_fn=search_generator(path=None, buffer=input_bytes), on_done=validate_img_fn)


def test_pdf_flow_mix_buffer():
    path = os.path.join(cur_dir, 'cats_are_awesome.pdf')
    with open(path, 'rb') as pdf:
        input_bytes = pdf.read()
    f = Flow().add(uses='PDFExtractorSegmenter', array_in_pb=True)
    with f:
        f.search(input_fn=search_generator(path=None, buffer=input_bytes), on_done=validate_mix_fn)
