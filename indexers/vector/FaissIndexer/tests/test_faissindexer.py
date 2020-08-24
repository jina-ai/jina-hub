import gzip
import os
import shutil

import numpy as np
from jina.executors.indexers import BaseIndexer

from .. import FaissIndexer

# fix the seed here
np.random.seed(500)
retr_idx = None
vec_idx = np.random.randint(0, high=100, size=[10])
vec = np.array(np.random.random([10, 10]), dtype=np.float32)
query = np.array(np.random.random([10, 10]), dtype=np.float32)
cur_dir = os.path.dirname(os.path.abspath(__file__))


def rm_files(file_paths):
    for file_path in file_paths:
        if os.path.exists(file_path):
            if os.path.isfile(file_path):
                os.remove(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path, ignore_errors=False, onerror=None)


def test_faiss_indexer():
    train_filepath = os.path.join(cur_dir, 'train.tgz')
    train_data = np.array(np.random.random([1024, 10]), dtype=np.float32)
    with gzip.open(train_filepath, 'wb', compresslevel=1) as f:
        f.write(train_data.tobytes())

    with FaissIndexer(index_filename='faiss.test.gz', index_key='IVF10,PQ2', train_filepath=train_filepath) as indexer:
        indexer.add(vec_idx, vec)
        indexer.save()
        assert os.path.exists(indexer.index_abspath)
        index_abspath = indexer.index_abspath
        save_abspath = indexer.save_abspath

    with BaseIndexer.load(save_abspath) as indexer:
        assert isinstance(indexer, FaissIndexer)
        idx, dist = indexer.query(query, top_k=4)
        global retr_idx
        if retr_idx is None:
            retr_idx = idx
        else:
            np.testing.assert_almost_equal(retr_idx, idx)
        assert idx.shape == dist.shape
        assert idx.shape == (10, 4)

    rm_files([index_abspath, save_abspath, train_filepath])
