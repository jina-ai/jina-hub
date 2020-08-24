import os
import shutil

import numpy as np
from jina.executors.indexers import BaseIndexer
from jina.executors.indexers.vector import NumpyIndexer

from .. import SptagIndexer

# fix the seed here
np.random.seed(500)
retr_idx = None
num_dimensions = 10
num_vectors = 30
num_queries = 20
top_k = 4
vec_idx = np.random.randint(0, high=100, size=[num_vectors])
vec = np.random.random([num_vectors, num_dimensions]).astype(np.float32)
query = np.random.random([num_queries, num_dimensions]).astype(np.float32)
cur_dir = os.path.dirname(os.path.abspath(__file__))


def rm_files(file_paths):
    for file_path in file_paths:
        if os.path.exists(file_path):
            if os.path.isfile(file_path):
                os.remove(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path, ignore_errors=False, onerror=None)


def test_sptagindexer():
    with SptagIndexer(index_filename='np.test.gz') as indexer:
        indexer.add(vec_idx, vec)
        indexer.save()
        assert os.path.exists(indexer.index_abspath)
        index_abspath = indexer.index_abspath
        save_abspath = indexer.save_abspath

    with BaseIndexer.load(indexer.save_abspath) as indexer:
        assert isinstance(indexer, SptagIndexer)
        idx, dist = indexer.query(query, top_k=top_k)
        global retr_idx
        if retr_idx is None:
            retr_idx = idx
        else:
            np.testing.assert_almost_equal(retr_idx, idx)
        assert idx.shape == dist.shape
        assert idx.shape == (num_queries, top_k)

    rm_files([index_abspath, save_abspath])


def test_sptag_wrap_indexer():
    with NumpyIndexer(index_filename='wrap-npidx.gz') as indexer:
        indexer.name = 'wrap-npidx'
        indexer.add(vec_idx, vec)
        indexer.save()
        index_abspath = indexer.index_abspath
        save_abspath = indexer.save_abspath

    with BaseIndexer.load_config(os.path.join(cur_dir, 'yaml/sptag-wrap.yml')) as indexer:
        assert isinstance(indexer, SptagIndexer)
        idx, dist = indexer.query(query, top_k=top_k)
        global retr_idx
        if retr_idx is None:
            retr_idx = idx
        else:
            np.testing.assert_almost_equal(retr_idx, idx)
        assert idx.shape == dist.shape
        assert idx.shape == (num_queries, top_k)

    rm_files([index_abspath, save_abspath])
