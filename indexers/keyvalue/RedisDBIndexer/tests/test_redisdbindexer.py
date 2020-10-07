import os
import shutil
import pytest

import jina.proto.jina_pb2 as jina_pb2
from google.protobuf.json_format import MessageToJson
from jina.executors.indexers import BaseIndexer
from jina.executors.metas import get_default_metas

from .. import RedisDBIndexer

@pytest.fixture(scope='function', autouse=True)
def metas(tmpdir):
    os.environ['TEST_WORKSPACE'] = str(tmpdir)	    
    metas = get_default_metas()	    
    metas['workspace'] = os.environ['TEST_WORKSPACE']
    yield metas
    del os.environ['TEST_WORKSPACE']

def test_redis_db_indexer(metas):
    def create_document(doc_id, text, weight, length):
        d = jina_pb2.Document()
        d.id = doc_id
        d.buffer = text.encode('utf8')
        d.weight = weight
        d.length = length
        return d

    #with indexer as idx:
    with RedisDBIndexer(metas=metas) as idx: 
        data = {
            'd1': MessageToJson(create_document(1, 'cat', 0.1, 3)),
            'd2': MessageToJson(create_document(2, 'dog', 0.2, 3)),
            'd3': MessageToJson(create_document(3, 'bird', 0.3, 3)),
        }
        idx.add(data)
        idx.touch()
        idx.save()
        save_abspath = idx.save_abspath
        index_abspath = idx.index_abspath
    assert os.path.exists(index_abspath)
    assert os.path.exists(save_abspath)

    with BaseIndexer.load(save_abspath) as searcher:
        doc = searcher.query('d2')
        assert doc.id == 2
        assert doc.length == 3


