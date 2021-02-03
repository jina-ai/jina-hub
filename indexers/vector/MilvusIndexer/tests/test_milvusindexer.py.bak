import os
import shutil

from jina.executors.indexers import BaseIndexer

from .. import MilvusIndexer


def rm_files(file_paths):
    for file_path in file_paths:
        if os.path.exists(file_path):
            if os.path.isfile(file_path):
                os.remove(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path, ignore_errors=False, onerror=None)


def test_milvus_indexer_save_and_load():
    with MilvusIndexer('localhost', 19530,
                       'collection', 'IVF', {'key': 'value'}) as indexer:
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

    rm_files([save_abspath])
