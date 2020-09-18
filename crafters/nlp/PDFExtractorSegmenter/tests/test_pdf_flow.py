from jina.flow import Flow
import os


from jina.flow import Flow
import os


def validate_text_fn(resp):
    expected_text = "A cat poem\nI love cats, I love every kind of cat,\nI just wanna hug all of them, but I can't," \
                    "\nI'm thinking about cats again\nI think about how cute they are\nAnd their whiskers and their " \
                    "nose\n"

    for d in resp.search.docs:
        assert expected_text == d.chunks[0].text



def test_pdf_flow():
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(cur_dir, 'cats_are_awesome_text.pdf')

    f = Flow().add(uses='PDFExtractorSegmenter')

    with f:
        f.search([path], output_fn=validate_text_fn)
