import os
import shutil

from jina import Document
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


def create_document(doc_id, text, weight, length):
    d = Document()
    d._document.id = doc_id
    d.buffer = text.encode('utf8')
    d.weight = weight
    d.length = length
    return d


def run_test(indexer):
    with indexer as idx:
        keys = iter([1, 2, 3])
        values = iter([
            MessageToJson(create_document('1', 'cat', 0.1, 3)),
            MessageToJson(create_document('2', 'dog', 0.2, 3)),
            MessageToJson(create_document('3', 'bird', 0.3, 3)),
        ])
        idx.add(keys, values)
        idx.touch()
        idx.save()
        save_abspath = idx.save_abspath
        index_abspath = idx.index_abspath
    assert os.path.exists(index_abspath)
    assert os.path.exists(save_abspath)

    with BaseIndexer.load(save_abspath) as searcher:
        doc = searcher.query(2)
        assert doc.buffer == b'dog'
        assert doc.length == 3

    rm_files([save_abspath, index_abspath])


def test_add_query():
    indexer = LevelDBIndexer(level='doc', index_filename='leveldb.db')
    run_test(indexer)


def test_load_yaml():
    from jina.executors import BaseExecutor
    indexer = BaseExecutor.load_config(os.path.join(cur_dir, 'yaml/test-leveldb.yml'))
    run_test(indexer)


def test_delete_query():
    indexer = LevelDBIndexer(level='doc', index_filename='leveldb.db')

    with indexer as idx:
        keys = iter([1, 2, 3])
        values = iter([
            MessageToJson(create_document('1', 'cat', 0.1, 3)),
            MessageToJson(create_document('2', 'dog', 0.2, 3)),
            MessageToJson(create_document('3', 'bird', 0.3, 3)),
        ])
        idx.add(keys, values)
        idx.delete(iter([2]))
        idx.touch()
        idx.save()
        save_abspath = idx.save_abspath
        index_abspath = idx.index_abspath
    assert os.path.exists(index_abspath)
    assert os.path.exists(save_abspath)

    with BaseIndexer.load(save_abspath) as searcher:
        doc = searcher.query(1)
        assert doc.buffer == b'cat'
        assert doc.length == 3
        doc = searcher.query(2)
        assert doc is None
        doc = searcher.query(3)
        assert doc.buffer == b'bird'
        assert doc.length == 3

