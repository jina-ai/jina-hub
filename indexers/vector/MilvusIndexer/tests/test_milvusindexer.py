import os

import pytest
from jina.executors.indexers import BaseIndexer
from jina.executors.metas import get_default_metas

from .. import MilvusIndexer
from ..milvusdbhandler import MilvusDBHandler


@pytest.fixture(scope='function', autouse=True)
def metas(tmpdir):
    os.environ['TEST_WORKSPACE'] = str(tmpdir)
    metas = get_default_metas()
    metas['workspace'] = os.environ['TEST_WORKSPACE']
    yield metas
    del os.environ['TEST_WORKSPACE']


def test_milvus_indexer_save_and_load(monkeypatch, metas):
    monkeypatch.setattr(MilvusDBHandler, 'connect', None)
    with MilvusIndexer('localhost', 19530,
                       'collection', 'IVF', {'key': 'value'},
                       metas=metas) as indexer:
        indexer.touch()
        indexer.save()
        assert os.path.exists(indexer.save_abspath)
        save_abspath = indexer.save_abspath

    with BaseIndexer.load(save_abspath) as indexer:
        assert isinstance(indexer, MilvusIndexer)
        assert indexer.host == 'localhost'
        assert indexer.port == 19530
        assert indexer.collection_name == 'collection'
        assert indexer.index_type == 'IVF'
        assert indexer.index_params['key'] == 'value'
