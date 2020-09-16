from .. import PDFExtractorSegmenter

expected_text = "A cat poem\nI love cats, I love every kind of cat,\nI just wanna hug all of them, but I can't," \
                    "\nI'm thinking about cats again\nI think about how cute they are\nAnd their whiskers and their " \
                    "nose\n "

def test_io_uri():
    crafter = PDFExtractorSegmenter()
    chunks = crafter.craft(uri='cats_are_awesome.pdf', buffer=None)
    print("expected_text ", expected_text)
    assert chunks[0]['text'] == expected_text
