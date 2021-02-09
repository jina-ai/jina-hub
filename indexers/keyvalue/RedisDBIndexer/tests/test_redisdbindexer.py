from pathlib import Path

import pytest
import redis

from jina import Document
from jina.executors import BaseExecutor
from jina.executors.indexers import BaseIndexer
from .. import RedisDBIndexer

cur_dir = Path(__file__).parent.absolute()


def create_document(doc_id, text, weight, length):
    d = Document()
    d.id = str(doc_id)
    d.buffer = text.encode('utf8')
    d.weight = weight
    d.length = length
    return d


def get_documents(ids):
    documents = [
        ['0', 'cat', 0.1, 3],
        ['1', 'dog', 0.2, 3],
        ['2', 'crow', 0.3, 4],
        ['3', 'pikachu', 0.4, 7],
        ['4', 'magikarp', 0.5, 8]
    ]
    id_to_document = {d[0]: d for d in documents}
    data = [
        (
            id_to_document[doc_id][0],
            id_to_document[doc_id],
            create_document(*id_to_document[doc_id])
        )
        for doc_id in ids
    ]
    return zip(*data)


def apply_action(idx, action):
    method, key_to_id = action
    with idx:
        keys = key_to_id.keys()
        ids = key_to_id.values()
        ids, documents, documents_proto = get_documents(ids)
        if method == 'add':
            idx.add(keys, [doc.SerializeToString() for doc in documents_proto])
        elif method == 'delete':
            idx.delete(keys)
        elif method == 'update':
            idx.update(keys, [doc.SerializeToString() for doc in documents_proto])
        else:
            print(f'method {method} is not implemented')
        idx.touch()
        idx.save()


def apply_actions(save_abspath, index_abspath, actions):
    for action in actions:
        idx = BaseIndexer.load(save_abspath)
        apply_action(idx, action)

    return save_abspath, index_abspath


def validate_positive_results(keys, documents, searcher: RedisDBIndexer):
    print('validate_positive_results')
    for key, query_doc in zip(keys, documents):
        result = searcher.query(key)
        result_doc = Document()
        result_doc.proto.ParseFromString(result)
        assert result_doc.id == str(query_doc[0])
        assert result_doc.buffer == query_doc[1].encode('utf8')
        assert round(result_doc.weight, 5) == query_doc[2]
        assert result_doc.length == query_doc[3]


def validate_negative_results(keys, searcher):
    for key in keys:
        result_docs = searcher.query(key)
        assert result_docs is None


def validate_results(save_abspath, results, negative_results):
    with BaseIndexer.load(save_abspath) as searcher:
        keys, ids = zip(*[[k, v] for k, v in results.items()])
        _, documents, _ = get_documents(ids)
        validate_positive_results(keys, documents, searcher)
        validate_negative_results(negative_results, searcher)


def get_indexers():
    # test construction from code
    indexer_1 = RedisDBIndexer(db=0)
    # test construction from yaml
    indexer_2 = BaseExecutor.load_config(str(cur_dir / 'yaml/test-redis.yml'))
    return indexer_1, indexer_2


def run_crud_test_exception_aware(actions, results, no_results, exception, mocker, tmpdir):
    # action is defined as (method, key, document_id)
    for indexer in get_indexers():
        r = redis.Redis(host='0.0.0.0', port=6379, db=0, socket_timeout=10)
        r.flushdb()
        indexer.workspace = tmpdir
        with indexer:
            indexer.touch()
            indexer.save()
        save_abspath = indexer.save_abspath
        index_abspath = indexer.index_abspath
        assert Path(index_abspath).exists()
        assert Path(save_abspath).exists()
        if exception is not None:
            with pytest.raises(exception):
                apply_actions(save_abspath, index_abspath, actions)
        else:
            apply_actions(save_abspath, index_abspath, actions)
            validate_results(save_abspath, results, no_results)


def test_basic_add(mocker, tmpdir):
    run_crud_test_exception_aware(
        [('add', {0: '0', 1: '1'}), ('add', {3: '3'})],
        {0: '0', 1: '1', 3: '3'},
        [2],
        None,
        mocker,
        tmpdir
    )


def test_add_existing_key(mocker, tmpdir):
    run_crud_test_exception_aware(
        [('add', {0: '0', 1: '1'}), ('add', {0: '3'})],
        {0: '3', 1: '1'},
        [2],
        None,
        mocker,
        tmpdir
    )


def test_update_existing_key(mocker, tmpdir):
    run_crud_test_exception_aware(
        [('add', {1: '1', 2: '2'}), ('update', {1: '4'})],
        {1: '4', 2: '2'},
        [0, 3],
        None,
        mocker,
        tmpdir
    )


def test_same_value(mocker, tmpdir):
    run_crud_test_exception_aware(
        [('add', {1: '1', 2: '2'}), ('update', {1: '2'})],
        {1: '2', 2: '2'},
        [0, 3],
        None,
        mocker,
        tmpdir
    )


def test_chain(mocker, tmpdir):
    run_crud_test_exception_aware(
        [('add', {0: '0', 1: '1'}), ('delete', {1: '1'}), ('add', {3: '3', 9: '4', 2: '4'}), ('delete', {0: '0', 2: '4'}),
         ('update', {3: '0'})],
        {9: '4', 3: '0'},
        [0, 1, 4],
        None,
        mocker,
        tmpdir
    )
