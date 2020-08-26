import os
import shutil

import jina.proto.jina_pb2 as jina_pb2
from google.protobuf.json_format import MessageToJson
from jina.executors.indexers import BaseIndexer

from .. import LevelDBIndexer

cur_dir = os.path.dirname(os.path.abspath(__file__))


def rm_files(file_paths):
    for file_path in file_paths:
        if os.path.exists(file_path):
            if os.path.isfile(file_path):
                os.remove(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path, ignore_errors=False, onerror=None)


def run_test(indexer):
    def create_document(doc_id, text, weight, length):
        d = jina_pb2.Document()
        d.id = doc_id
        d.buffer = text.encode('utf8')
        d.weight = weight
        d.length = length
        return d

    with indexer as idx:
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

    rm_files([save_abspath, index_abspath])


def test_add_query():
    indexer = LevelDBIndexer(level='doc', index_filename='leveldb.db')
    run_test(indexer)


def test_load_yaml():
    from jina.executors import BaseExecutor
    indexer = BaseExecutor.load_config(os.path.join(cur_dir, 'yaml/test-leveldb.yml'))
    run_test(indexer)
