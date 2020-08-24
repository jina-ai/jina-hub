import os
import shutil

import numpy as np
from jina.executors.indexers import BaseIndexer

from .. import NGTIndexer

# fix the seed here
np.random.seed(500)
retr_idx = None
vec_idx = np.random.randint(0, high=100, size=[1, 10])
vec = np.random.random([10, 10])
query = np.array(np.random.random([10, 10]), dtype=np.float32)


def rm_files(file_paths):
    for file_path in file_paths:
        if os.path.exists(file_path):
            if os.path.isfile(file_path):
                os.remove(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path, ignore_errors=False, onerror=None)


def test_simple_ngt():
    import ngtpy
    path = '/tmp/ngt-index'
    dimension, queries, top_k, batch_size, num_batch = 10, 3, 5, 8, 3

    ngtpy.create(path=path, dimension=dimension, distance_type='L2')
    _index = ngtpy.Index(path=path)
    for i in range(num_batch):
        _index.batch_insert(np.random.random((batch_size, dimension)), num_threads=4)
    assert os.path.exists(path)

    idx = []
    dist = []
    for key in np.random.random((queries, dimension)):
        results = _index.search(key, size=top_k, epsilon=0.1)
        index_k = []
        distance_k = []
        [(index_k.append(result[0]), distance_k.append(result[1])) for result in results]
        idx.append(index_k)
        dist.append(distance_k)

    idx = np.array(idx)
    dist = np.array(dist)

    assert idx.shape == dist.shape
    assert idx.shape == (queries, top_k)


def test_ngt_indexer():
    with NGTIndexer(index_filename='ngt.test.gz') as indexer:
        indexer.add(vec_idx, vec)
        indexer.save()
        assert os.path.exists(indexer.index_abspath)
        index_abspath = indexer.index_abspath
        save_abspath = indexer.save_abspath

    with BaseIndexer.load(save_abspath) as indexer:
        assert isinstance(indexer, NGTIndexer)
        idx, dist = indexer.query(query, top_k=4)
        global retr_idx
        if retr_idx is None:
            retr_idx = idx
        else:
            np.testing.assert_almost_equal(retr_idx, idx)
        assert idx.shape == dist.shape
        assert idx.shape == (10, 4)

    rm_files([index_abspath, save_abspath])
