__copyright__ = "Copyright (c) 2021 Jina AI Limited. All rights reserved."
__license__ = "Apache-2.0"

import os
import pytest

import numpy as np

from .. import ZarrIndexer
from jina.executors.indexers import BaseIndexer
from jina.executors.indexers.vector import NumpyIndexer
from jina.executors.metas import get_default_metas

np.random.seed(500)
retr_idx = None
num_data = 10
num_dim = 64
num_query = 100
keys = np.random.randint(0, high=100, size=[num_data]).astype(str)
np.random.shuffle(keys)
vec = np.random.random([num_data, num_dim])
query = np.array(np.random.random([num_query, num_dim]), dtype=np.float32)


@pytest.fixture(scope='function', autouse=True)
def metas(tmpdir):
    os.environ['TEST_WORKSPACE'] = str(tmpdir)
    metas = get_default_metas()
    metas['workspace'] = os.environ['TEST_WORKSPACE']
    yield metas
    del os.environ['TEST_WORKSPACE']


def test_zarr_indexer(metas):
    with ZarrIndexer(index_filename='test.zarr', metric='euclidean', metas=metas) as indexer:
        indexer.add(keys, vec)
        indexer.save()
        assert os.path.exists(indexer.index_abspath)
        assert indexer._raw_ndarray.shape == vec.shape
        assert 'default' in indexer.write_handler.array_keys()
        save_abspath = indexer.save_abspath

    with ZarrIndexer.load(save_abspath) as indexer:
        assert isinstance(indexer, ZarrIndexer)
        idx, dist = indexer.query(query, top_k=4)
        assert idx.shape == dist.shape
        assert idx.shape == (num_query, 4)


def test_zarr_indexer_known(metas):
    vectors = np.array([[1, 1, 1],
                        [10, 10, 10],
                        [100, 100, 100],
                        [1000, 1000, 1000]])
    keys = np.array([4, 5, 6, 7]).reshape(-1, 1)
    with ZarrIndexer(index_filename='test.zarr', metric='euclidean', metas=metas) as indexer:
        indexer.add(keys, vectors)
        indexer.save()
        assert os.path.exists(indexer.index_abspath)
        save_abspath = indexer.save_abspath

    queries = np.array([[1, 1, 1],
                        [10, 10, 10],
                        [100, 100, 100],
                        [1000, 1000, 1000]])
    with ZarrIndexer.load(save_abspath) as indexer:
        assert isinstance(indexer, NumpyIndexer)
        idx, dist = indexer.query(queries, top_k=2)
        np.testing.assert_equal(idx, np.array([['4', '5'], ['5', '4'], ['6', '5'], ['7', '6']]))
        assert idx.shape == dist.shape
        assert idx.shape == (4, 2)
        np.testing.assert_equal(indexer.query_by_key(['7', '4']), vectors[[3, 0]])


def test_zarr_indexer_known_big(metas):
    """Let's try to have some real test. We will have an index with 10k vectors of random values between 5 and 10.
     We will change tweak some specific vectors that we expect to be retrieved at query time. We will tweak vector
     at index [0, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000], this will also be the query vectors.
     Then the keys will be assigned shifted to test the proper usage of `int2ext_id` and `ext2int_id`
    """
    vectors = np.random.uniform(low=5.0, high=10.0, size=(10000, 1024)).astype('float32')

    queries = np.empty((10, 1024))
    for idx in range(0, 10000, 1000):
        array = idx * np.ones((1, 1024))
        queries[int(idx / 1000)] = array
        vectors[idx] = array

    keys = np.arange(10000, 20000).reshape(-1, 1).astype(str)

    with ZarrIndexer(index_filename='test.zarr', metric='euclidean', metas=metas) as indexer:
        indexer.add(keys, vectors)
        indexer.save()
        assert os.path.exists(indexer.index_abspath)
        save_abspath = indexer.save_abspath

    with BaseIndexer.load(save_abspath) as indexer:
        assert isinstance(indexer, ZarrIndexer)
        idx, dist = indexer.query(queries, top_k=1)
        np.testing.assert_equal(idx, np.array(
            [['10000'], ['11000'], ['12000'], ['13000'], ['14000'], ['15000'], ['16000'], ['17000'], ['18000'], ['19000']]))
        assert idx.shape == dist.shape
        assert idx.shape == (10, 1)
        np.testing.assert_equal(indexer.query_by_key(['10000', '15000']), vectors[[0, 5000]])
