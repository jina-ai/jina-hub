from typing import io

from jina.drivers.helper import array2pb
from jina.proto import jina_pb2
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
            blob = d.chunks[chunk].blob
            assert blob.shape[1], blob.shape[0] == img.size


def validate_mix_fn(resp):
    for d in resp.search.docs:
        for chunk in range(len(d.chunks) - 1):
            img = Image.open(os.path.join(cur_dir, f'test_img_{chunk}.jpg'))
            blob = d.chunks[chunk].blob
            assert blob.shape[1], blob.shape[0] == img.size

        assert expected_text == d.chunks[2].text


def search_generator(path: str, buffer: bytes):
    d = jina_pb2.Document()
    if buffer:
        d.buffer = buffer
    if path:
        d.uri = path
    yield d


def test_pdf_flow():
    path = os.path.join(cur_dir, 'cats_are_awesome_text.pdf')

    f = Flow().add(uses='PDFExtractorSegmenter', array_in_pb=True)

    with f:
        f.search(input_fn=search_generator(path=path, buffer=None), output_fn=validate_text_fn)


def test_pdf_flow_buffer():
    path = os.path.join(cur_dir, 'cats_are_awesome.pdf')
    with open(path, 'rb') as pdf:
        input_bytes = pdf.read()

    f = Flow().add(uses='PDFExtractorSegmenter', array_in_pb=True)

    with f:
        f.search(input_fn=search_generator(path=None, buffer=input_bytes), output_fn=validate_mix_fn)

