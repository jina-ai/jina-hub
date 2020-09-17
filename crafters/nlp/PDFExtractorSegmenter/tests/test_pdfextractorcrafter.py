from .. import PDFExtractorSegmenter
from PIL import Image
import os

expected_text = "A cat poem\nI love cats, I love every kind of cat,\nI just wanna hug all of them, but I can't," \
                "\nI'm thinking about cats again\nI think about how cute they are\nAnd their whiskers and their " \
                "nose\n "


def test_io_uri():
    crafter = PDFExtractorSegmenter()
    chunks = crafter.craft(uri='cats_are_awesome.pdf', buffer=None)

    for i in range(len(chunks)):
        print("list(chunks[", i, "])[0] ", list(chunks[i])[0])
        if list(chunks[i])[0] == 'text':
            # Check test
            assert chunks[i]['text'] == expected_text
        elif list(chunks[i])[0] == 'blob':
            # Check image
            cur_dir = os.path.dirname(os.path.abspath(__file__))
            img = Image.open(os.path.join(cur_dir, '../test_img.jpg'))
            blob = chunks[i]['blob']
            assert img.width == blob.width
            assert img.height == blob.height

