__copyright__ = "Copyright (c) 2021 Jina AI Limited. All rights reserved."
__license__ = "Apache-2.0"

import numpy as np
import pytest

from .. import TikaExtractor


def input_bytes():
    with open('cats_are_awesome.pdf', 'rb') as pdf:
        i_bytes = pdf.read()
    return i_bytes


@pytest.mark.parametrize('uri, buffer', [
    (np.stack(['cats_are_awesome.pdf', 'cats_are_awesome.pdf']), [None, None]),
    ([None, None], np.stack([input_bytes(), input_bytes()]))
])
def test_extraction(uri, buffer):
    tika_extractor = TikaExtractor()
    crafted_docs = tika_extractor.craft(uri, buffer)
    assert len(crafted_docs) == 2
    for crafted_doc in crafted_docs:
        assert len(crafted_doc['text']) > 20


@pytest.mark.parametrize('uri, buffer', [
    ('cats_are_awesome.pdf', None),
    (None, input_bytes())
])
def test_extraction_single(uri, buffer):
    tika_extractor = TikaExtractor()
    crafted_doc = tika_extractor.craft(uri, buffer)
    assert len(crafted_doc['text']) > 20


@pytest.mark.parametrize('uri, buffer', [
    ('cats_are_awesome.pdf', None),
    (None, input_bytes())
])
def test_extraction_single(uri, buffer):
    tika_extractor = TikaExtractor()
    crafted_doc = tika_extractor.craft(uri=uri, buffer=buffer)
    assert len(crafted_doc['text']) > 20
