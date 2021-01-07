import os
import pytest

import numpy as np
from jina.executors.indexers import BaseIndexer
from jina.executors.indexers.vector import NumpyIndexer
from jina.executors.metas import get_default_metas

from .. import AnnoyIndexer

# fix the seed here
np.random.seed(500)
retr_idx = None
vec_idx = np.random.randint(0, high=100, size=[10])
vec = np.random.random([10, 10])
query = np.array(np.random.random([10, 10]), dtype=np.float32)
cur_dir = os.path.dirname(os.path.abspath(__file__))


@pytest.fixture(scope='function', autouse=True)
def metas(tmpdir):
    os.environ['TEST_WORKSPACE'] = str(tmpdir)
    metas = get_default_metas()
    metas['workspace'] = os.environ['TEST_WORKSPACE']
    yield metas
    del os.environ['TEST_WORKSPACE']


def test_annoy_wrap_indexer(metas):
    with NumpyIndexer(index_filename='wrap-npidx.gz', metas=metas) as indexer:
        indexer.name = 'wrap-npidx'
        indexer.add(vec_idx, vec)

    with BaseIndexer.load_config(os.path.join(cur_dir, 'yaml/annoy-wrap.yml')) as indexer:
        assert isinstance(indexer, AnnoyIndexer)
        idx, dist = indexer.query(query, top_k=4)
        assert idx.shape == dist.shape
        assert idx.shape == (10, 4)


def test_simple_annoy():
    from annoy import AnnoyIndex
    _index = AnnoyIndex(5, 'angular')
    for j in range(3):
        _index.add_item(j, np.random.random((5,)))
    _index.build(4)
    idx1, _ = _index.get_nns_by_vector(np.random.random((5,)), 3, include_distances=True)
    assert len(idx1) == 3


def test_annoy_indexer(metas):
    with AnnoyIndexer(index_filename='annoy.test.gz', metas=metas) as indexer:
        indexer.add(vec_idx, vec)
        indexer.save()
        assert os.path.exists(indexer.index_abspath)
        save_abspath = indexer.save_abspath

    with BaseIndexer.load(save_abspath) as indexer:
        assert isinstance(indexer, AnnoyIndexer)
        idx, dist = indexer.query(query, top_k=4)
        assert idx.shape == dist.shape
        assert idx.shape == (10, 4)


def test_annoy_indexer_with_no_search_k(metas):
    with AnnoyIndexer(index_filename='annoy.test.gz', search_k=0, metas=metas) as indexer:
        indexer.add(vec_idx, vec)
        indexer.save()
        assert os.path.exists(indexer.index_abspath)

        save_abspath = indexer.save_abspath

    with BaseIndexer.load(save_abspath) as indexer:
        assert isinstance(indexer, AnnoyIndexer)
        idx, dist = indexer.query(query, top_k=4)
        # search_k is 0, so no tree is searched for
        assert idx.shape == dist.shape
        assert idx.shape == (10, 0)


def test_annoy_indexer_known(metas):
    vectors = np.array([[1, 1, 1],
                        [10, 10, 10],
                        [100, 100, 100],
                        [1000, 1000, 1000]])
    keys = np.array([4, 5, 6, 7]).reshape(-1, 1)
    with AnnoyIndexer(index_filename='annoy.test.gz', metas=metas) as indexer:
        indexer.add(keys, vectors)
        indexer.save()
        assert os.path.exists(indexer.index_abspath)
        save_abspath = indexer.save_abspath

    queries = np.array([[1, 1, 1],
                        [10, 10, 10],
                        [100, 100, 100],
                        [1000, 1000, 1000]])
    with BaseIndexer.load(save_abspath) as indexer:
        assert isinstance(indexer, AnnoyIndexer)
        idx, dist = indexer.query(queries, top_k=2)
        np.testing.assert_equal(idx, np.array([[4, 5], [5, 4], [6, 5], [7, 6]]))
        assert idx.shape == dist.shape
        assert idx.shape == (4, 2)
        np.testing.assert_equal(indexer.query_by_id([7, 4]), vectors[[3, 0]])


def test_annoy_indexer_known_update_delete(metas):
    vectors = np.array([[1, 1, 1],
                        [10, 10, 10],
                        [100, 100, 100],
                        [1000, 1000, 1000]])
    keys = np.array([4, 5, 6, 7]).reshape(-1, 1)
    with AnnoyIndexer(index_filename='annoy.test.gz', metas=metas) as indexer:
        indexer.add(keys, vectors)
        indexer.save()
        assert os.path.exists(indexer.index_abspath)
        save_abspath = indexer.save_abspath

    queries = np.array([[1, 1, 1],
                        [10, 10, 10],
                        [100, 100, 100],
                        [1000, 1000, 1000]])
    with BaseIndexer.load(save_abspath) as indexer:
        assert isinstance(indexer, AnnoyIndexer)
        idx, dist = indexer.query(queries, top_k=2)
        np.testing.assert_equal(idx, np.array([[4, 5], [5, 4], [6, 5], [7, 6]]))
        assert idx.shape == dist.shape
        assert idx.shape == (4, 2)
        np.testing.assert_equal(indexer.query_by_id([7, 4]), vectors[[3, 0]])

    # update
    with BaseIndexer.load(save_abspath) as indexer:
        indexer.update([4], np.array([[200, 200, 200]]))
        indexer.save()
        assert indexer.size == 4

    with BaseIndexer.load(save_abspath) as indexer:
        assert isinstance(indexer, AnnoyIndexer)
        idx, dist = indexer.query(queries, top_k=3)
        np.testing.assert_equal(idx, np.array([[5, 6, 4], [5, 6, 4], [6, 5, 4], [7, 4, 6]]))
        assert idx.shape == dist.shape
        assert idx.shape == (4, 3)

    # delete
    with BaseIndexer.load(save_abspath) as indexer:
        indexer.delete([4])
        indexer.save()
        assert indexer.size == 3

    with BaseIndexer.load(save_abspath) as indexer:
        assert isinstance(indexer, AnnoyIndexer)
        idx, dist = indexer.query(queries, top_k=2)
        np.testing.assert_equal(idx, np.array([[5, 6], [5, 6], [6, 5], [7, 6]]))
        assert idx.shape == dist.shape
        assert idx.shape == (4, 2)


def test_annoy_indexer_known_big(metas):
    """Let's try to have some real test. We will have an index with 10k vectors of random values between 5 and 10.
     We will change tweak some specific vectors that we expect to be retrieved at query time. We will tweak vector
     at index [0, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000], this will also be the query vectors.
     Then the keys will be assigned shifted to test the proper usage of `int2ext_id` and `ext2int_id`
    """
    vectors = np.random.uniform(low=5.0, high=10.0, size=(10000, 1024))

    queries = np.empty((10, 1024))
    for idx in range(0, 10000, 1000):
        array = idx * np.ones((1, 1024))
        queries[int(idx / 1000)] = array
        vectors[idx] = array

    keys = np.arange(10000, 20000).reshape(-1, 1)
    with AnnoyIndexer(index_filename='annoy.test.gz', n_trees=5000, metas=metas) as indexer:
        indexer.add(keys, vectors)
        indexer.save()
        assert os.path.exists(indexer.index_abspath)
        save_abspath = indexer.save_abspath

    with BaseIndexer.load(save_abspath) as indexer:
        assert isinstance(indexer, AnnoyIndexer)
        idx, dist = indexer.query(queries, top_k=1)
        print(f' idx {idx}')
        np.testing.assert_equal(idx, np.array(
            [[10000], [11000], [12000], [13000], [14000], [15000], [16000], [17000], [18000], [19000]]))
        assert idx.shape == dist.shape
        assert idx.shape == (10, 1)
        np.testing.assert_equal(indexer.query_by_id([10000, 15000]), vectors[[0, 5000]])
