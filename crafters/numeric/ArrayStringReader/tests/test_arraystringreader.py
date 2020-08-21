import numpy as np

from .. import ArrayStringReader


def test_arraystringreader():
    """here is my test code

    https://docs.pytest.org/en/stable/getting-started.html#create-your-first-test
    """
    size = 8
    sample_array = np.random.rand(size).astype('float32')
    text = ','.join([str(x) for x in sample_array])

    reader = ArrayStringReader()
    crafted_doc = reader.craft(text, 0)

    assert crafted_doc['blob'].shape[0] == size
    np.testing.assert_array_equal(crafted_doc['blob'], sample_array)
