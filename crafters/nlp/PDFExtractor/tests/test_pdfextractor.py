from .. import PDFTextExtractor


expected_text = "A cat poem\nI love cats, I love every kind of cat,\nI just wanna hug all of them, but I can't," \
                    "\nI'm thinking about cats again\nI think about how cute they are\nAnd their whiskers and their " \
                    "nose\n"

def test_io_uri():
    crafter = PDFTextExtractor()
    text = crafter.craft(uri='cats_are_awesome.pdf', buffer=None)
    assert text == expected_text

def test_io_buffer():
    with open('cats_are_awesome.pdf', 'rb') as pdf:
        input_bytes = pdf.read()
    crafter = PDFTextExtractor()
    text = crafter.craft(uri=None, buffer=input_bytes)
    assert text == expected_text
