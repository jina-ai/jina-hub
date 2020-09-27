from .. import TikaExtractor


def test_extraction_uri():
    tika_extractor = TikaExtractor()
    text = tika_extractor.craft(uri='cats_are_awesome.pdf', buffer=None)
    assert len(text) > 20


def test_extraction_bytes():
    tika_extractor = TikaExtractor()
    with open('cats_are_awesome.pdf', 'rb') as pdf:
        input_bytes = pdf.read()
    text = tika_extractor.craft(uri=None, buffer=input_bytes)
    assert len(text) > 20
