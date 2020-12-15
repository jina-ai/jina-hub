import gzip
import os
import pytest

import numpy as np
from jina.executors.indexers import BaseIndexer
from jina.executors.metas import get_default_metas

from .. import FaissIndexer

# fix the seed here
np.random.seed(500)
retr_idx = None
vec_idx = np.random.randint(0, high=100, size=[10])
vec = np.array(np.random.random([10, 10]), dtype=np.float32)
query = np.array(np.random.random([10, 10]), dtype=np.float32)
cur_dir = os.path.dirname(os.path.abspath(__file__))


@pytest.fixture(scope='function', autouse=True)
def metas(tmpdir):
    os.environ['TEST_WORKSPACE'] = str(tmpdir)
    metas = get_default_metas()
    metas['workspace'] = os.environ['TEST_WORKSPACE']
    yield metas
    del os.environ['TEST_WORKSPACE']


def test_faiss_indexer(metas):
    train_filepath = os.path.join(os.environ['TEST_WORKSPACE'], 'train.tgz')
    train_data = np.array(np.random.random([1024, 10]), dtype=np.float32)
    with gzip.open(train_filepath, 'wb', compresslevel=1) as f:
        f.write(train_data.tobytes())

    with FaissIndexer(index_filename='faiss.test.gz', index_key='IVF10,PQ2', train_filepath=train_filepath,
                      metas=metas) as indexer:
        indexer.add(vec_idx, vec)
        indexer.save()
        assert os.path.exists(indexer.index_abspath)
        save_abspath = indexer.save_abspath

    with BaseIndexer.load(save_abspath) as indexer:
        assert isinstance(indexer, FaissIndexer)
        idx, dist = indexer.query(query, top_k=4)
        assert idx.shape == dist.shape
        assert idx.shape == (10, 4)


def test_faiss_indexer_known(metas):
    vectors = np.array([[1, 1, 1],
                        [10, 10, 10],
                        [100, 100, 100],
                        [1000, 1000, 1000]], dtype=np.float32)
    keys = np.array([4, 5, 6, 7]).reshape(-1, 1)

    train_filepath = os.path.join(os.environ['TEST_WORKSPACE'], 'train.tgz')
    train_data = vectors
    with gzip.open(train_filepath, 'wb', compresslevel=1) as f:
        f.write(train_data.tobytes())

    with FaissIndexer(index_filename='faiss.test.gz', index_key='Flat', train_filepath=train_filepath,
                      metas=metas) as indexer:
        indexer.add(keys, vectors)
        indexer.save()
        assert os.path.exists(indexer.index_abspath)
        save_abspath = indexer.save_abspath

    queries = np.array([[1, 1, 1],
                        [10, 10, 10],
                        [100, 100, 100],
                        [1000, 1000, 1000]], dtype=np.float32)
    with BaseIndexer.load(save_abspath) as indexer:
        assert isinstance(indexer, FaissIndexer)
        idx, dist = indexer.query(queries, top_k=2)
        np.testing.assert_equal(idx, np.array([[4, 5], [5, 4], [6, 5], [7, 6]]))
        assert idx.shape == dist.shape
        assert idx.shape == (4, 2)
        np.testing.assert_equal(indexer.query_by_id([7, 4]), vectors[[3, 0]])


def test_faiss_indexer_known_update_delete(metas):
    vectors = np.array([[1, 1, 1],
                        [10, 10, 10],
                        [100, 100, 100],
                        [1000, 1000, 1000]], dtype=np.float32)
    keys = np.array([4, 5, 6, 7]).reshape(-1, 1)

    train_filepath = os.path.join(os.environ['TEST_WORKSPACE'], 'train.tgz')
    train_data = vectors
    with gzip.open(train_filepath, 'wb', compresslevel=1) as f:
        f.write(train_data.tobytes())

    with FaissIndexer(index_filename='faiss.test.gz', index_key='Flat', train_filepath=train_filepath,
                      metas=metas) as indexer:
        indexer.add(keys, vectors)
        indexer.save()
        assert os.path.exists(indexer.index_abspath)
        save_abspath = indexer.save_abspath

    queries = np.array([[1, 1, 1],
                        [10, 10, 10],
                        [100, 100, 100],
                        [1000, 1000, 1000]], dtype=np.float32)
    with BaseIndexer.load(save_abspath) as indexer:
        assert isinstance(indexer, FaissIndexer)
        idx, dist = indexer.query(queries, top_k=2)
        np.testing.assert_equal(idx, np.array([[4, 5], [5, 4], [6, 5], [7, 6]]))
        assert idx.shape == dist.shape
        assert idx.shape == (4, 2)
        np.testing.assert_equal(indexer.query_by_id([7, 4]), vectors[[3, 0]])

    # update
    with BaseIndexer.load(save_abspath) as indexer:
        indexer.update([4], np.array([[200, 200, 200]], dtype=np.float32))
        indexer.save()
        assert indexer.size == 4

    with BaseIndexer.load(save_abspath) as indexer:
        assert isinstance(indexer, FaissIndexer)
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
        assert isinstance(indexer, FaissIndexer)
        idx, dist = indexer.query(queries, top_k=2)
        np.testing.assert_equal(idx, np.array([[5, 6], [5, 6], [6, 5], [7, 6]]))
        assert idx.shape == dist.shape
        assert idx.shape == (4, 2)


def test_faiss_indexer_known_big(metas):
    """Let's try to have some real test. We will have an index with 10k vectors of random values between 5 and 10.
     We will change tweak some specific vectors that we expect to be retrieved at query time. We will tweak vector
     at index [0, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000], this will also be the query vectors.
     Then the keys will be assigned shifted to test the proper usage of `int2ext_id` and `ext2int_id`
    """
    vectors = np.random.uniform(low=5.0, high=10.0, size=(10000, 1024)).astype('float32')

    queries = np.empty((10, 1024), dtype=np.float32)
    for idx in range(0, 10000, 1000):
        array = idx * np.ones((1, 1024), dtype=np.float32)
        queries[int(idx / 1000)] = array
        vectors[idx] = array

    train_filepath = os.path.join(os.environ['TEST_WORKSPACE'], 'train.tgz')
    train_data = vectors
    with gzip.open(train_filepath, 'wb', compresslevel=1) as f:
        f.write(train_data.tobytes())

    keys = np.arange(10000, 20000).reshape(-1, 1)

    with FaissIndexer(index_filename='faiss.test.gz', index_key='Flat', train_filepath=train_filepath,
                      metas=metas) as indexer:
        indexer.add(keys, vectors)
        indexer.save()
        assert os.path.exists(indexer.index_abspath)
        save_abspath = indexer.save_abspath

    with BaseIndexer.load(save_abspath) as indexer:
        assert isinstance(indexer, FaissIndexer)
        idx, dist = indexer.query(queries, top_k=1)
        np.testing.assert_equal(idx, np.array(
            [[10000], [11000], [12000], [13000], [14000], [15000], [16000], [17000], [18000], [19000]]))
        assert idx.shape == dist.shape
        assert idx.shape == (10, 1)
        np.testing.assert_equal(indexer.query_by_id([10000, 15000]), vectors[[0, 5000]])
