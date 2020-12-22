import os

import pytest
import numpy as np

from milvus import Milvus, DataType
from jina.executors.indexers import BaseIndexer
from jina.executors.metas import get_default_metas

from .. import MilvusIndexer
from ..milvusdbhandler import MilvusDBHandler

cur_dir = os.path.dirname(os.path.abspath(__file__))

port = 19530
host = '0.0.0.0'
collection_name = 'test_collection'


@pytest.fixture
def collection():
    with Milvus(host, str(port)) as client:
        if not client.has_collection(collection_name):
            param = {
                'fields': [
                    {
                        'name': 'embedding',
                        'type': DataType.FLOAT_VECTOR,
                        'params': {'dim': 8}
                    }
                ],
                'segment_row_limit': 4096,
                'auto_id': False
                }
            client.create_collection(param)
    yield
    with Milvus(host, str(port)) as client:
        client.drop_collection(collection_name)


@pytest.fixture(scope='function', autouse=True)
def metas(tmpdir):
    os.environ['TEST_WORKSPACE'] = str(tmpdir)
    metas = get_default_metas()
    metas['workspace'] = os.environ['TEST_WORKSPACE']
    yield metas
    del os.environ['TEST_WORKSPACE']


def test_milvus_indexer_save_and_load(metas):
    with MilvusIndexer(host, port,
                       collection_name, 'IVF', {'key': 'value'},
                       metas=metas) as indexer:
        indexer.touch()
        indexer.save()
        assert os.path.exists(indexer.save_abspath)
        save_abspath = indexer.save_abspath

    with BaseIndexer.load(save_abspath) as indexer:
        assert isinstance(indexer, MilvusIndexer)
        assert indexer.host == host
        assert indexer.port == port
        assert indexer.collection_name == collection_name
        assert indexer.index_type == 'IVF'
        assert indexer.index_params['key'] == 'value'


def test_milvusdbhandler_simple(collection):
    vectors = np.array([[1, 1, 1],
                        [10, 10, 10],
                        [100, 100, 100],
                        [1000, 1000, 1000]])
    keys = np.array([0, 1, 2, 3]).reshape(-1, 1)
    with MilvusDBHandler(host, port, collection_name) as db:
        db.insert(keys, vectors)
        dist, idx = db.search(vectors, 2)
        dist = np.array(dist)
        idx = np.array(idx)
        assert idx.shape == dist.shape
        assert idx.shape == (4, 2)
        np.testing.assert_equal(idx, np.array([[0, 1], [1, 0], [2, 1], [3, 2]]))


def test_milvusdbhandler_build(collection):
    vectors = np.array([[1, 1, 1],
                        [10, 10, 10],
                        [100, 100, 100],
                        [1000, 1000, 1000]])
    keys = np.array([0, 1, 2, 3]).reshape(-1, 1)
    with MilvusDBHandler(host, port, collection_name) as db:
        db.insert(keys, vectors)
        db.build_index(index_type='IVF,Flat', index_params={'nlist': 2})

        dist, idx = db.search(vectors, 2, {'nprobe': 2})
        dist = np.array(dist)
        idx = np.array(idx)
        assert idx.shape == dist.shape
        assert idx.shape == (4, 2)
        np.testing.assert_equal(idx, np.array([[0, 1], [1, 0], [2, 1], [3, 2]]))


def test_milvus_indexer(collection):
    vectors = np.array([[1, 1, 1],
                        [10, 10, 10],
                        [100, 100, 100],
                        [1000, 1000, 1000]])
    keys = np.array([0, 1, 2, 3]).reshape(-1, 1)
    with MilvusIndexer(host=host, port=port,
                       collection_name=collection_name, index_type='IVF,Flat',
                       index_params={'nlist': 2}) as indexer:
        indexer.add(keys, vectors)
        idx, dist = indexer.query(vectors, 2, search_params={'nprobe': 2})
        dist = np.array(dist)
        idx = np.array(idx)
        assert idx.shape == dist.shape
        assert idx.shape == (4, 2)
        np.testing.assert_equal(idx, np.array([[0, 1], [1, 0], [2, 1], [3, 2]]))
