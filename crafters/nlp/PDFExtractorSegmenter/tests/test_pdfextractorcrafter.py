from .. import PDFExtractorSegmenter
from PIL import Image
import os

expected_text = "A cat poem\nI love cats, I love every kind of cat,\nI just wanna hug all of them, but I can't," \
                    "\nI'm thinking about cats again\nI think about how cute they are\nAnd their whiskers and their " \
                    "nose\n "

def test_io_uri():
    crafter = PDFExtractorSegmenter()
    chunks = crafter.craft(uri='cats_are_awesome.pdf', buffer=None)

    # Check image
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    img = Image.open(os.path.join(cur_dir, '../test_img.jpg'))
    blob = chunks[0]['blob']
    assert img.width == blob.width
    assert img.height == blob.height

    # Check test
    assert chunks[2]['text'] == expected_text