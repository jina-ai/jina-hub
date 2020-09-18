from jina.proto import jina_pb2
from jina.drivers.helper import array2pb
from jina.flow import Flow
import os


def validate_text_fn(resp):
    expected_text = "A cat poem\nI love cats, I love every kind of cat,\nI just wanna hug all of them, but I can't," \
                    "\nI'm thinking about cats again\nI think about how cute they are\nAnd their whiskers and their " \
                    "nose\n"

    for d in resp.search.docs:
        assert expected_text == d.chunks[0].text

def search_generator(path):
    d = jina_pb2.Document()
    d.uri = path
    yield d


def test_pdf_flow():
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(cur_dir, 'cats_are_awesome.pdf')

    f = Flow().add(uses='PDFExtractorSegmenter')

    with f:
        f.search(search_generator(path), output_fn=print)
