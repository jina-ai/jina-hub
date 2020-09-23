import os
import shutil

import zarr
import numpy as np

from .. import ZarrIndexer
from jina.executors.indexers import BaseIndexer
from jina.executors.indexers.vector import NumpyIndexer

np.random.seed(500)
retr_idx = None
num_data = 10000
num_dim = 64
num_query = 100
vec_idx = np.arange(num_data)
np.random.shuffle(vec_idx)
vec = np.random.random([num_data, num_dim])
query = np.array(np.random.random([num_query, num_dim]), dtype=np.float32)


def rm_files(file_paths):
    for file_path in file_paths:
        if os.path.exists(file_path):
            if os.path.isfile(file_path):
                os.remove(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path, ignore_errors=False, onerror=None)
                

def test_zarr_indexer():
    with ZarrIndexer(index_filename='test.zarr') as indexer:
        indexer.add(vec_idx, vec)
        indexer.save()
        assert os.path.exists(indexer.index_abspath)
        assert indexer.raw_ndarray.shape == vec.shape
        assert 'default' in indexer.write_handler.array_keys()
        index_abspath = indexer.index_abspath
        save_abspath = indexer.save_abspath

    with ZarrIndexer.load(save_abspath) as indexer:
        assert isinstance(indexer, NumpyIndexer)
        idx, dist = indexer.query(query, top_k=4)
        global retr_idx
        if retr_idx is None:
            retr_idx = idx
        else:
            np.testing.assert_almost_equal(retr_idx, idx)
        assert idx.shape == dist.shape
        assert idx.shape == (num_query, 4)
    rm_files([index_abspath, save_abspath])
    

def test_zarr_indexer_known():
    vectors = np.array([[1, 1, 1],
                        [10, 10, 10],
                        [100, 100, 100],
                        [1000, 1000, 1000]])
    keys = np.array([4, 5, 6, 7]).reshape(-1, 1)
    with ZarrIndexer(index_filename='test.zarr') as indexer:
        indexer.add(keys, vectors)
        indexer.save()
        assert os.path.exists(indexer.index_abspath)
        index_abspath = indexer.index_abspath
        save_abspath = indexer.save_abspath

    queries = np.array([[1, 1, 1],
                        [10, 10, 10],
                        [100, 100, 100],
                        [1000, 1000, 1000]])
    with ZarrIndexer.load(save_abspath) as indexer:
        assert isinstance(indexer, NumpyIndexer)
        idx, dist = indexer.query(queries, top_k=2)
        np.testing.assert_equal(idx, np.array([[4, 5], [5, 4], [6, 5], [7, 6]]))
        assert idx.shape == dist.shape
        assert idx.shape == (4, 2)
        np.testing.assert_equal(indexer.query_by_id([7, 4]), vectors[[3, 0]])

    rm_files([index_abspath, save_abspath])
