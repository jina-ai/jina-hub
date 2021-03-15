__copyright__ = "Copyright (c) 2021 Jina AI Limited. All rights reserved."
__license__ = "Apache-2.0"

import pytest

import numpy as np

from .. import ArrayStringReader


@pytest.mark.parametrize('single', [False, True])
def test_arraystringreader(single):
    _size = 8
    sample_array = np.random.rand(_size).astype('float32')
    text = ','.join([str(x) for x in sample_array])

    reader = ArrayStringReader()
    crafted_docs = reader.craft(text if single else np.stack([text, text]))

    if single:
        assert crafted_docs['blob'].shape[0] == _size
        np.testing.assert_array_equal(crafted_docs['blob'], sample_array)
    else:
        assert len(crafted_docs) == 2
        for crafted_doc in crafted_docs:
            assert crafted_doc['blob'].shape[0] == _size
            np.testing.assert_array_equal(crafted_doc['blob'], sample_array)


def test_arraystringreader_keywords():
    _size = 8
    sample_array = np.random.rand(_size).astype('float32')
    text = ','.join([str(x) for x in sample_array])

    reader = ArrayStringReader()
    crafted_doc = reader.craft(text=text)

    assert crafted_doc['blob'].shape[0] == _size
    np.testing.assert_array_equal(crafted_doc['blob'], sample_array)
