from .. import TikaExtractor


def test_extraction():
    tika_extractor = TikaExtractor()
    extraction = tika_extractor.craft(uri='cats_are_awesome.pdf')
    assert 'text' in extraction
    assert 'metadata' in extraction
    assert len(extraction['text']) > 20
