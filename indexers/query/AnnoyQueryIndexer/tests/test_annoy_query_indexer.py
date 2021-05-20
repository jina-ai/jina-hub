__copyright__ = "Copyright (c) 2021 Jina AI Limited. All rights reserved."
__license__ = "Apache-2.0"

import os

import numpy as np
import pytest
from jina.executors.indexers.dump import export_dump_streaming
from jina.executors.metas import get_default_metas

from .. import AnnoyQueryIndexer

# fix the seed here
np.random.seed(500)
vec_idx = np.random.randint(0, high=100, size=[10]).astype(str)
vec = np.random.random([10, 10])

query = np.array(np.random.random([10, 10]), dtype=np.float32)


@pytest.fixture(scope='function', autouse=True)
def metas(tmpdir):
    metas = get_default_metas()
    metas['workspace'] = tmpdir
    metas['dump_path'] = os.path.join(tmpdir, 'dump')
    yield metas


def test_query_annoy_indexer_known_big(metas):
    vectors = np.random.uniform(low=5.0, high=10.0, size=(10000, 1024))
    # notice how it only reads from dump
    queries = np.empty((10, 1024))
    for idx in range(0, 10000, 1000):
        array = idx * np.ones((1, 1024))
        queries[int(idx / 1000)] = array
        vectors[idx] = array

    keys = np.arange(10000, 20000).astype(str)

    export_dump_streaming(
        path=metas['dump_path'],
        shards=1,
        size=len(keys),
        data=zip(keys, vectors, [b'NOTUSED' for _ in range(len(keys))])
    )

    # notice how index_filename is not used
    with AnnoyQueryIndexer(metas=metas) as indexer:
        idx, dist = indexer.query(queries, top_k=1)
        print(idx)
        np.testing.assert_equal(idx, np.array(
            [[10000], [11000], [12000], [13000], [14000], [15000], [16000], [17000], [18000], [19000]]).astype(
            str)), idx
        assert idx.shape == dist.shape
        assert idx.shape == (10, 1)
        np.testing.assert_equal(indexer.query_by_key(['10000', '15000']), vectors[[0, 5000]])


def test_query_annoy(metas):
    # notice how it only reads from dump
    export_dump_streaming(
        path=metas['dump_path'],
        shards=1,
        size=len(vec_idx),
        data=zip(vec_idx, vec, [b'NOTUSED' for _ in range(len(vec_idx))])
    )

    # notice how index_filename is not used
    with AnnoyQueryIndexer(search_k=0, metas=metas) as indexer:
        assert indexer.size == len(vec_idx)
        idx, dist = indexer.query(query, top_k=4)
        # search_k is 0, so no tree is searched for
        assert idx.shape == dist.shape
        assert idx.shape == (10, 0)
