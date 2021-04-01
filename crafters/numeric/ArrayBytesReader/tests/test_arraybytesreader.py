__copyright__ = "Copyright (c) 2021 Jina AI Limited. All rights reserved."
__license__ = "Apache-2.0"

import pytest

import numpy as np

from .. import ArrayBytesReader


@pytest.mark.parametrize('as_type', ['float32', 'int'])
@pytest.mark.parametrize('single', [True, False])
def test_bytes_reader(single, as_type):
    _size = 8
    sample_array = np.random.rand(_size).astype(as_type)
    array_bytes = sample_array.tobytes()

    reader = ArrayBytesReader(as_type=as_type)

    crafted_docs = reader.craft(array_bytes if single else [array_bytes, array_bytes])
    if single:
        assert crafted_docs['blob'].shape[0] == _size
        np.testing.assert_array_equal(crafted_docs['blob'], sample_array)
    else:
        assert len(crafted_docs) == 2
        for crafted_doc in crafted_docs:
            assert crafted_doc['blob'].shape[0] == _size
            np.testing.assert_array_equal(crafted_doc['blob'], sample_array)


@pytest.mark.parametrize('data_type, reader_type', [('float32', 'float64')])
def test_bytes_reader_wrong_type(data_type, reader_type):
    _size = 8
    sample_array = np.random.rand(_size).astype(data_type)
    array_bytes = sample_array.tobytes()

    reader = ArrayBytesReader(as_type=reader_type)
    crafted_doc = reader.craft(array_bytes)

    assert crafted_doc['blob'].shape[0] == _size / 2


@pytest.mark.parametrize('as_type', ['float32', 'int'])
def test_bytes_reader_keyword(as_type):
    _size = 8
    sample_array = np.random.rand(_size).astype(as_type)
    array_bytes = sample_array.tobytes()

    reader = ArrayBytesReader(as_type=as_type)

    crafted_doc = reader.craft(buffer=array_bytes)
    assert crafted_doc['blob'].shape[0] == _size
    np.testing.assert_array_equal(crafted_doc['blob'], sample_array)
