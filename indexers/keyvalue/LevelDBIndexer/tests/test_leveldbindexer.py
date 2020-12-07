import os
import shutil

import pytest
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
    d._document.id = (str(doc_id) * 16)[:16]
    d.buffer = text.encode('utf8')
    d.weight = weight
    d.length = length
    return d


def get_documents(ids):
    documents = [
        [0, 'cat', 0.1, 3],
        [1, 'dog', 0.2, 3],
        [2, 'crow', 0.3, 4],
        [3, 'pikachu', 0.4, 7],
        [4, 'magikarp', 0.5, 8]
    ]
    id_to_document = {d[0]: d for d in documents}
    data = [
        (id_to_document[id][0], id_to_document[id], MessageToJson(create_document(*id_to_document[id])).encode('utf-8'))
        for id in ids
    ]
    return zip(*data)


def apply_action(idx, action):
    method, key_to_id = action
    with idx:
        keys = key_to_id.keys()
        ids = key_to_id.values()
        ids, documents, documents_bytes = get_documents(ids)
        if method == 'add':
            idx.add(keys, documents_bytes)
        elif method == 'delete':
            idx.delete(keys)
        elif method == 'update':
            idx.update(keys, documents_bytes)
        else:
            print(f'method {method} is not implemented')
        idx.touch()
        idx.save()


def apply_actions(save_abspath, index_abspath, actions):
    for action in actions:
        idx = BaseIndexer.load(save_abspath)
        apply_action(idx, action)

    return save_abspath, index_abspath


def validate_positive_results(keys, documents, searcher):
    for key, query_doc in zip(keys, documents):
        result_doc = searcher.query(key)
        assert result_doc.id == str(query_doc[0]) * 16
        assert result_doc.buffer == query_doc[1].encode('utf8')
        assert round(result_doc.weight, 5) == query_doc[2]
        assert result_doc.length == query_doc[3]


def validate_negative_results(keys, searcher):
    for key in keys:
        result_doc = searcher.query(key)
        assert result_doc is None


def validate_results(save_abspath, results, negative_results):
    with BaseIndexer.load(save_abspath) as searcher:
        if results:
            keys, ids = zip(*[[k, v] for k, v in results.items()])
            _, documents, _ = get_documents(ids)
            validate_positive_results(keys, documents, searcher)
        if negative_results:
            validate_negative_results(negative_results, searcher)


def run_crud_test_for_scenario(indexer, actions, results, no_results):
    exception = None
    try:
        with indexer:
            indexer.touch()
            indexer.save()
        save_abspath = indexer.save_abspath
        index_abspath = indexer.index_abspath
        assert os.path.exists(index_abspath)
        assert os.path.exists(save_abspath)
        apply_actions(save_abspath, index_abspath, actions)
        validate_results(save_abspath, results, no_results)
    except Exception as e:
        exception = e
    finally:
        rm_files([save_abspath, index_abspath])
        if exception is not None:
            raise exception


def run_crud_test(actions, results, no_results):
    # test construction from code
    indexer = LevelDBIndexer(level='doc', index_filename='leveldb.db')
    run_crud_test_for_scenario(indexer, actions, results, no_results)

    # test construction from yaml
    from jina.executors import BaseExecutor
    indexer = BaseExecutor.load_config(os.path.join(cur_dir, 'yaml/test-leveldb.yml'))
    run_crud_test_for_scenario(indexer, actions, results, no_results)


def run_crud_test_exception_aware(actions, results, no_results, exception, mocker):

    # action is defined as (method, key, document_id)
    if exception is not None:
        with pytest.raises(exception):
            run_crud_test(actions, results, no_results)
    else:
        global validate_positive_results, validate_negative_results
        validate_positive_results = mocker.Mock(wraps=validate_positive_results)
        validate_negative_results = mocker.Mock(wraps=validate_negative_results)
        run_crud_test(actions, results, no_results)
        validate_positive_results.assert_called()
        validate_negative_results.assert_called()


def test_basic_add(mocker):
    run_crud_test_exception_aware(
        [('add', {0: 0, 1: 1}), ('add', {3: 3})],
        {0: 0, 1: 1, 3: 3},
        [2],
        None,
        mocker
    )


def test_add_existing_key(mocker):
    run_crud_test_exception_aware(
        [('add', {0: 0, 1: 1}), ('add', {0: 3})],
        {0: 3, 1: 1},
        [2],
        None,
        mocker
    )


def test_update_existing_key(mocker):
    run_crud_test_exception_aware(
        [('add', {1: 1, 2: 2}), ('update', {1: 4})],
        {1: 4, 2: 2},
        [0, 3],
        None,
        mocker
    )


def test_update_non_existing_key(mocker):
    run_crud_test_exception_aware(
        [('update', {1: 4})],
        {},
        [0, 3],
        KeyError,
        mocker
    )


def test_update_existing_and_non_existing_key(mocker):
    run_crud_test_exception_aware(
        [('add', {1: 1, 2: 2}), ('update', {1: 4, 3: 4})],
        {},
        [0, 3],
        KeyError,
        mocker
    )


def test_same_value(mocker):
    run_crud_test_exception_aware(
        [('add', {1: 1, 2: 2}), ('update', {1: 2})],
        {1: 2, 2: 2},
        [0, 3],
        None,
        mocker
    )


def test_chain(mocker):
    run_crud_test_exception_aware(
        [('add', {0: 0, 1: 1}), ('delete', {1: 1}), ('add', {3: 3, 9: 4, 2: 4}), ('delete', {0: 0, 2: 4}),
         ('update', {3: 0})],
        {9: 4, 3: 0},
        [0, 1, 4],
        None,
        mocker
    )
