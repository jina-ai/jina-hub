__copyright__ = "Copyright (c) 2021 Jina AI Limited. All rights reserved."
__license__ = "Apache-2.0"

from .. import TikaExtractor


def test_extraction_uri():
    tika_extractor = TikaExtractor()
    extract = tika_extractor.craft(uri='cats_are_awesome.pdf', buffer=None)
    assert len(extract['text']) > 20


def test_extraction_bytes():
    tika_extractor = TikaExtractor()
    with open('cats_are_awesome.pdf', 'rb') as pdf:
        input_bytes = pdf.read()
    extract = tika_extractor.craft(uri=None, buffer=input_bytes)
    assert len(extract['text']) > 20
