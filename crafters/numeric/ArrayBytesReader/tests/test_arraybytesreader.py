import numpy as np

from .. import ArrayBytesReader


def test_bytes_reader():
    size = 8
    sample_array = np.random.rand(size).astype('float32')
    array_bytes = sample_array.tobytes()

    reader = ArrayBytesReader()
    crafted_doc = reader.craft(array_bytes, 0)

    assert crafted_doc['blob'].shape[0] == size
    np.testing.assert_array_equal(crafted_doc['blob'], sample_array)

def test_bytes_reader_int_type():
    size = 8
    sample_array = np.random.rand(size).astype('int')
    array_bytes = sample_array.tobytes()

    reader = ArrayBytesReader(as_type='int')
    crafted_doc = reader.craft(array_bytes, 0)

    assert crafted_doc['blob'].shape[0] == size
    np.testing.assert_array_equal(crafted_doc['blob'], sample_array)

def test_bytes_reader_wrong_type():
    size = 8
    sample_array = np.random.rand(size).astype('float32')
    array_bytes = sample_array.tobytes()

    reader = ArrayBytesReader(as_type='float64')
    crafted_doc = reader.craft(array_bytes, 0)

    assert crafted_doc['blob'].shape[0] == size / 2
